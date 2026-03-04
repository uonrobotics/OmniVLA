import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import TwistStamped

import os
import sys
import numpy as np
import cv2
import torch
from typing import Tuple, Dict, Type
from PIL import Image

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP

# OmniVLA Original 필수 임포트 (경로가 올바른지 확인하세요)
sys.path.insert(0, '..')
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE
from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor

# ===============================================================
# 추론 설정 클래스
# ===============================================================
class InferenceConfig:
    vla_path: str = "./omnivla-original"  # 베이스 모델(백본) 경로
    num_images_in_input: int = 2

# ===============================================================
# Select Modality
# ===============================================================
pose_goal = False
satellite = False
image_goal = False
lan_prompt = True

# ===============================================================
# Utility Functions
# ===============================================================
def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")

# ===============================================================
# ROS2 Node Class
# ===============================================================
class OmniVLAROSNode(Node):
    def __init__(self):
        super().__init__("omnivla_original_node")
        
        # 1. 설정 초기화
        self.cfg = InferenceConfig()
        
        # 2. 모델 로드
        self.define_model()
        
        # 3. 상태 변수 및 주행 설정
        self.current_image_pil = None
        self.lan_inst_prompt = "Search right to find marker 2, approach it, and stop."
        self.goal_image_pil = Image.open("./inference/goal_image.jpg").convert("RGB")
        
        # 4. ROS 통신 설정
        self.sub_img = self.create_subscription(
            CompressedImage, "/act/input/front_rgb/compressed", self.cb_image, 10)
        self.pub_cmd = self.create_publisher(
            TwistStamped, "/act/output/cmd_vel", 10)
        
        # 3Hz 주기로 추론 실행
        self.create_timer(0.33, self.inference_loop)
        self.get_logger().info(f"🚀 OmniVLA Original Node Started")

    def define_model(self):
        self.cfg.vla_path = self.cfg.vla_path.rstrip("/")
        print(f"Loading OpenVLA Model `{self.cfg.vla_path}`")

        # GPU setup
        self.device_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device_id)
        torch.cuda.empty_cache()

        print(
            "Detected constants:\n"
            f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
            f"\tACTION_DIM: {ACTION_DIM}\n"
            f"\tPOSE_DIM: {POSE_DIM}\n"
            f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
        )

        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
        
        # Load processor and VLA
        self.processor = AutoProcessor.from_pretrained(self.cfg.vla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self.device_id) #            trust_remote_code=True,
        
        self.vla.vision_backbone.set_num_images_in_input(self.cfg.num_images_in_input)
        self.vla.to(dtype=torch.bfloat16, device=self.device_id)

        # Load finetuned pose_projector
        self.pose_projector = ProprioProjector(
            llm_dim=self.vla.llm_dim, 
            proprio_dim=POSE_DIM
        )
        count_parameters(self.pose_projector, "pose_projector")
        p_path = "/home/lds/model/omnivla-original--210000_chkpt+lan_only+64batch/pose_projector--210000_checkpoint.pt"
        # p_path = "/home/lds/workspace/OmniVLA/omnivla-original/proprio_projector--120000_checkpoint.pt"
        print(f"Loading checkpoint: {p_path}")
        p_state = torch.load(p_path, map_location=self.device_id)
        p_state = remove_ddp_in_checkpoint(p_state)
        self.pose_projector.load_state_dict(p_state)
        self.pose_projector = self.pose_projector.to(self.device_id)

        # Load finetuned action_head
        self.action_head = L1RegressionActionHead_idcat(
            input_dim=self.vla.llm_dim, 
            hidden_dim=self.vla.llm_dim, 
            action_dim=ACTION_DIM
        )
        count_parameters(self.action_head, "action_head")
        a_path = "/home/lds/model/omnivla-original--210000_chkpt+lan_only+64batch/action_head--210000_checkpoint.pt"
        # a_path = "/home/lds/workspace/OmniVLA/omnivla-original/action_head--120000_checkpoint.pt"
        print(f"Loading checkpoint: {a_path}")
        a_state = torch.load(a_path, map_location=self.device_id)
        a_state = remove_ddp_in_checkpoint(a_state)
        self.action_head.load_state_dict(a_state)
        self.action_head = self.action_head.to(torch.bfloat16).to(self.device_id)

        # Get number of vision patches
        self.num_patches = (self.vla.vision_backbone.get_num_patches() * self.vla.vision_backbone.get_num_images_in_input())
        self.num_patches += 1 #for goal pose

        # Create Action Tokenizer
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)

    def cb_image(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.current_image_pil = Image.fromarray(cv_img_rgb)

    # ----------------------------
    # Transform Data to Dataset Format
    # ----------------------------
    def transform_datatype(self, inst_obj, actions, goal_pose_cos_sin,
                           current_image_PIL, goal_image_PIL, prompt_builder, action_tokenizer,
                           base_tokenizer, image_transform, predict_stop_token=True):
        IGNORE_INDEX = -100
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        if inst_obj == "xxxx":
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {inst_obj}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]

        prompt_builder = prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = torch.tensor(base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX

        pixel_values_current = image_transform(current_image_PIL)
        pixel_values_goal = image_transform(goal_image_PIL)
        dataset_name = "lelan"

        return dict(
            pixel_values_current=pixel_values_current,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=torch.as_tensor(actions),
            goal_pose=goal_pose_cos_sin,
            img_PIL=current_image_PIL,
            inst=inst_obj,
        )
    
    # ----------------------------
    # Custom Collator
    # ----------------------------
    def collator_custom(self, instances, model_max_length, pad_token_id, padding_side="right", pixel_values_dtype=torch.float32):
        IGNORE_INDEX = -100
        input_ids = pad_sequence([inst["input_ids"] for inst in instances], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([inst["labels"] for inst in instances], batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, :model_max_length], labels[:, :model_max_length]
        attention_mask = input_ids.ne(pad_token_id)

        pixel_values = [inst["pixel_values_current"] for inst in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [inst["dataset_name"] for inst in instances]
        else:
            dataset_names = None

        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_goal" in instances[0]:
                pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_goal)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type: {type(pixel_values)}")

        actions = torch.stack([torch.from_numpy(np.copy(inst["actions"])) for inst in instances])
        goal_pose = torch.stack([torch.from_numpy(np.copy(inst["goal_pose"])) for inst in instances])

        output = dict(
            pixel_values=pixel_values.to(),
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            goal_pose=goal_pose,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output
    
    # ----------------------------
    # Data Transformer for OmniVLA
    # ----------------------------
    def data_transformer_omnivla(self, current_image_PIL, lan_inst, goal_image_PIL, goal_pose_loc_norm,
                                 prompt_builder, action_tokenizer, processor):
        actions = np.random.rand(8, 4)  # dummy actions
        goal_pose_cos_sin = goal_pose_loc_norm

        batch_data = self.transform_datatype(
            lan_inst, actions, goal_pose_cos_sin,
            current_image_PIL, goal_image_PIL,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
        )

        batch = self.collator_custom(
            instances=[batch_data],
            model_max_length=processor.tokenizer.model_max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            padding_side="right"
        )
        return batch

    # ----------------------------
    # Run Forward Pass
    # ----------------------------
    def run_forward_pass(self, vla, action_head, noisy_action_projector, pose_projector,
                         batch, action_tokenizer, device_id, use_l1_regression, use_diffusion,
                         use_film, num_patches, compute_diffusion_l1=False,
                         num_diffusion_steps_train=None, mode="vali", idrun=0) -> Tuple[torch.Tensor, Dict[str, float]]:

        metrics = {}
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

        # Determine modality
        if satellite and not lan_prompt and not pose_goal and not image_goal:
            modality_id = torch.as_tensor([0], dtype=torch.float32)
        elif satellite and not lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([1], dtype=torch.float32)
        elif satellite and not lan_prompt and not pose_goal and image_goal:
            modality_id = torch.as_tensor([2], dtype=torch.float32)
        elif satellite and not lan_prompt and pose_goal and image_goal:
            modality_id = torch.as_tensor([3], dtype=torch.float32)
        elif not satellite and not lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([4], dtype=torch.float32)
        elif not satellite and not lan_prompt and pose_goal and image_goal:
            modality_id = torch.as_tensor([5], dtype=torch.float32)
        elif not satellite and not lan_prompt and not pose_goal and image_goal:
            modality_id = torch.as_tensor([6], dtype=torch.float32)
        elif not satellite and lan_prompt and not pose_goal and not image_goal:
            modality_id = torch.as_tensor([7], dtype=torch.float32)
        elif not satellite and lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([8], dtype=torch.float32)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                modality_id=modality_id.to(torch.bfloat16).to(device_id),
                labels=batch["labels"].to(device_id),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id),
                proprio_projector=pose_projector,
                noisy_actions=noisy_actions if use_diffusion else None,
                noisy_action_projector=noisy_action_projector if use_diffusion else None,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
                use_film=use_film,
            )

        # Prepare data for metrics
        ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
         
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)

        with torch.no_grad():
            predicted_actions = action_head.predict_action(actions_hidden_states, modality_id.to(torch.bfloat16).to(device_id))                                 

        # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
        return predicted_actions, modality_id

    def inference_loop(self):
        if self.current_image_pil is None: return

        # Dummy goal pose loc 
        # 좌표 기반 주행이 아니므로 이 값에 크게 영향이 없어야 함
        #   - 이 값은 "현재 위치가 목표 지점이고 각도 차이도 0이다"라는 정보임 
        goal_pose_loc_norm = np.array([0.0, 0.0, 1.0, 0.0])
        
        # Prepare batch
        batch = self.data_transformer_omnivla(
            self.current_image_pil,
            self.lan_inst_prompt,
            self.goal_image_pil,
            goal_pose_loc_norm,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )

        # Run forward pass
        actions, modality_id = self.run_forward_pass(
            vla=self.vla.eval(),
            action_head=self.action_head.eval(),
            noisy_action_projector=None,
            pose_projector=self.pose_projector.eval(),
            batch=batch,
            action_tokenizer=self.action_tokenizer,
            device_id=self.device_id,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_patches=self.num_patches,
            compute_diffusion_l1=False,
            num_diffusion_steps_train=None,
            mode="train",
        )

        waypoints = actions.float().cpu().numpy()
        metric_waypoint_spacing = 0.5

        # Select waypoint
        waypoint_select = 4
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint

        # PD controller
        EPS = 1e-8
        DT = 1 / 3
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * clip_angle(np.arctan2(hy, hx)) / DT
        elif np.abs(dx) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)
        else:
            linear_vel_value = dx / DT
            angular_vel_value = np.arctan(dy / dx) / DT

        linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
        angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)

        # Velocity limitation
        maxv, maxw = 0.3, 0.3
        if np.abs(linear_vel_value) <= maxv:
            if np.abs(angular_vel_value) <= maxw:
                linear_vel_value_limit = linear_vel_value
                angular_vel_value_limit = angular_vel_value
            else:
                rd = linear_vel_value / angular_vel_value
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        else:
            if np.abs(angular_vel_value) <= 0.001:
                linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                angular_vel_value_limit = 0.0
            else:
                rd = linear_vel_value / angular_vel_value
                if np.abs(rd) >= maxv / maxw:
                    linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                    angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)
                else:
                    linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                    angular_vel_value_limit = maxw * np.sign(angular_vel_value)

        print("linear angular", linear_vel_value_limit, angular_vel_value_limit)
        self.publish_command(linear_vel_value_limit, angular_vel_value_limit)

    def publish_command(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.pub_cmd.publish(msg)

def main():
    rclpy.init()
    node = OmniVLAROSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()