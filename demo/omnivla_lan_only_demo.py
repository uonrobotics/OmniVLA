import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import TwistStamped

import os
import sys
import time
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Optional, Tuple
from peft import PeftModel  # LoRA 로드를 위해 필수

# 프로젝트 패키지 경로 설정 (사용자 환경에 맞춰 수정)
sys.path.insert(0, '..')
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask

from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor

# ===============================================================
# 1. 설정 클래스 (체크포인트 경로 확인 필수)
# ===============================================================
class InferenceConfig:
    # 기본 모델 경로 (HuggingFace ID 혹은 로컬 경로)
    vla_path: str = "./omnivla-original" 
    
    # LoRA 어댑터 및 체크포인트가 저장된 디렉토리
    # 예: /path/to/run--210000_chkpt/
    checkpoint_dir: str = "/nas/sujinkim/model/goto/v1/omnivla-original--210000_chkpt+lan_only+64batch"
    resume_step: int = 210000
    num_images_in_input: int = 2

# ===============================================================
# 2. 유틸리티 함수
# ===============================================================
def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def load_checkpoint_file(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt")) and module_name == "pose_projector":
        module_name = "proprio_projector"
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return remove_ddp_in_checkpoint(state_dict)

# ===============================================================
# 3. ROS2 추론 노드 클래스
# ===============================================================
class OmniVLALoraInferenceNode(Node):
    def __init__(self):
        super().__init__("omnivla_lora_node")
        
        self.cfg = InferenceConfig()
        self.device_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 모델 초기화
        self.setup_model()
        
        # 상태 변수
        self.lan_inst_prompt = "Search right to find marker 1, approach it, and stop."
        self.metric_waypoint_spacing = 0.0625 # 0.125 * 0.5 (학습 기준)
        self.current_image_pil = None
        
        # ROS 통신
        self.sub_img = self.create_subscription(
            CompressedImage, "/act/input/front_rgb/compressed", self.cb_image, 10)
        self.pub_cmd = self.create_publisher(
            TwistStamped, "/act/output/cmd_vel", 10)
        
        # 타이머 (3Hz 주행)
        self.timer = self.create_timer(0.33, self.inference_loop)
        self.get_logger().info("OmnIVLA LoRA Inference Node Ready.")

    def setup_model(self):
        torch.cuda.empty_cache()
        from transformers import BitsAndBytesConfig
        
        # lora_adapter 폴더가 실제로는 전체 모델 체크포인트임
        model_path = os.path.join(self.cfg.checkpoint_dir, "lora_adapter")
        self.get_logger().info(f"Loading Fine-tuned Model from {model_path}...")
        
        # 1. 레지스터 등록 (동일)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
        
        # 2. 프로세서는 상위 폴더나 모델 폴더에서 로드
        self.processor = AutoProcessor.from_pretrained(self.cfg.checkpoint_dir, trust_remote_code=True)

        # 3. 4-bit 양자화 설정 (12GB VRAM 필수)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # 4. 모델 로드 (PeftModel 대신 직접 로드)
        # 4개의 분할된 safetensors 파일을 자동으로 합쳐서 로드합니다.
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map={"": 0}, 
            trust_remote_code=True
        )
        
        # 5. 추가 모듈 (Projector, Head) 로드
        # 이 파일들은 lora_adapter 바깥(상위 폴더)에 있으므로 cfg.checkpoint_dir 사용
        self.pose_projector = ProprioProjector(llm_dim=self.vla.llm_dim, proprio_dim=POSE_DIM)
        self.action_head = L1RegressionActionHead_idcat(input_dim=self.vla.llm_dim, hidden_dim=self.vla.llm_dim, action_dim=ACTION_DIM)
        
        self.pose_projector.load_state_dict(load_checkpoint_file("pose_projector", self.cfg.checkpoint_dir, self.cfg.resume_step))
        self.action_head.load_state_dict(load_checkpoint_file("action_head", self.cfg.checkpoint_dir, self.cfg.resume_step))

        self.pose_projector.to(torch.bfloat16).to(self.device_id).eval()
        self.action_head.to(torch.bfloat16).to(self.device_id).eval()
        
        try:
            self.vla.eval()
        except:
            pass

        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.num_patches = self.vla.vision_backbone.get_num_patches() * self.cfg.num_images_in_input + 1
        self.get_logger().info("Successfully loaded sharded model shards in 4-bit.")

    def cb_image(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.current_image_pil = Image.fromarray(cv_img_rgb)

    def inference_loop(self):
        if self.current_image_pil is None: return

        # 배치 데이터 구성
        dummy_pose = np.zeros(4)
        batch = self.prepare_batch(self.current_image_pil, self.lan_inst_prompt, dummy_pose)
        modality_id = torch.as_tensor([7], dtype=torch.float32).to(self.device_id)

        # 모델 추론 (No Grad)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.vla(
                input_ids=batch["input_ids"].to(self.device_id),
                attention_mask=batch["attention_mask"].to(self.device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(self.device_id),
                modality_id=modality_id.to(torch.bfloat16).to(self.device_id),
                labels=batch["labels"].to(self.device_id),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(self.device_id),
                proprio_projector=self.pose_projector,
                use_film=False,
            )

            # 액션 토큰 위치의 Hidden States 추출
            last_hidden = output.hidden_states[-1]
            text_hidden = last_hidden[:, self.num_patches:-1]
            gt_tokens = batch["labels"][:, 1:].to(self.device_id)
            mask = get_current_action_mask(gt_tokens) | get_next_actions_mask(gt_tokens)
            
            act_hidden = text_hidden[mask].reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)
            pred_actions = self.action_head.predict_action(act_hidden, modality_id.to(torch.bfloat16).to(self.device_id))

        # 궤적 해석 및 속도 제어
        waypoints = pred_actions.float().cpu().numpy()[0]
        v, w = self.calculate_pd_control(waypoints[4]) # 4번째 웨이포인트(중기 목표)
        
        self.publish_cmd(v, w)

    def prepare_batch(self, cur_img, inst, goal_pose):
        builder = PurePromptBuilder("openvla")
        builder.add_turn("human", f"What action should the robot take to {inst}?")
        builder.add_turn("gpt", "")
        
        ids = torch.tensor(self.processor.tokenizer(builder.get_prompt(), add_special_tokens=True).input_ids)
        pixel = self.processor.image_processor.apply_transform(cur_img)
        pixel_cat = torch.cat((pixel, pixel), dim=0) # OmniVLA 2-image input

        return {
            "input_ids": ids.unsqueeze(0),
            "attention_mask": ids.ne(self.processor.tokenizer.pad_token_id).unsqueeze(0),
            "pixel_values": pixel_cat.unsqueeze(0),
            "labels": ids.clone().unsqueeze(0),
            "goal_pose": torch.as_tensor(goal_pose, dtype=torch.float32).unsqueeze(0)
        }

    def calculate_pd_control(self, waypoint):
        z_norm, x_norm, cos_y, sin_y = waypoint
        target_z = z_norm * self.metric_waypoint_spacing
        target_x = -x_norm * self.metric_waypoint_spacing
        
        DT = 0.33
        target_yaw = -np.arctan2(sin_y, cos_y)
        
        # 선속도/각속도 계산
        dist = np.sqrt(target_z**2 + target_x**2)
        v = dist / DT
        w = target_yaw / DT
        
        # 하드웨어 보호를 위한 제한
        v = np.clip(v, 0.0, 0.4)
        w = np.clip(w, -0.5, 0.5)
        return v, w

    def publish_cmd(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.pub_cmd.publish(msg)
        self.get_logger().info(f"VLA CMD -> v:{v:.2f} w:{w:.2f}")

def main():
    rclpy.init()
    node = OmniVLALoraInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()