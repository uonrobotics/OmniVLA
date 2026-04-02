# ===============================================================
# [ OMNIVLA CLIENT ]
# 역할:
#   - IsaacSim server에서 pose/image 관측 받기
#   - OmniVLA 추론 수행
#   - (linear, angular) 계산
#   - ROS2 cmd_vel bridge로 action 전송
# ===============================================================

import base64
import io
import json
import math
import os
import socket
import time
import sys
import select
from typing import Optional, Tuple, Type

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoImageProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
    POSE_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)

# ===============================================================
# Goal definitions
# ===============================================================
goal_poses = {
    "forklift": (-2.71, -2.45143, 2.45),
    "marker1": (-20.11, 7.0, 1.57),
    "marker2": (-15.36, 7.0, 1.57),
    "marker3": (-10.47, 7.0, 1.57),
    "marker4": (-5.47, 7.0, 1.57),
    "marker5": (-0.6, 7.0, 1.57),
    "pallet": (0.54, -13.29, 0.31),
}

goal_image_paths = {
    "forklift": "./goal_img/forklift.png",
    "marker1": "./goal_img/marker1.png",
    "marker2": "./goal_img/marker2.png",
    "marker3": "./goal_img/marker3.png",
    "marker4": "./goal_img/marker4.png",
    "marker5": "./goal_img/marker5.png",
    "pallet": "./goal_img/pallet.png",
}

WAYPOINT_SPACING = 0.25

# modality flags
pose_goal = True
satellite = False
image_goal = True
lan_prompt = False


# ===============================================================
# Socket helpers
# ===============================================================
class JsonSocketClient:
    """
    Request/response client for simulator server.
    Each request expects one JSON line response.
    """

    def __init__(self, host: str = "192.168.0.180", port: int = 8765):
        self.sock = socket.create_connection((host, port))
        self.buffer = b""

    def request(self, payload: dict) -> dict:
        self.sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))

        while b"\n" not in self.buffer:
            data = self.sock.recv(10_000_000)
            if not data:
                raise RuntimeError("Simulator server disconnected.")
            self.buffer += data

        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


class JsonLineSender:
    """
    Fire-and-forget sender for cmd_vel bridge.
    Sends one JSON line per action.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8766):
        self.sock = socket.create_connection((host, port))

    def send(self, payload: dict):
        self.sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


# ===============================================================
# Model helpers
# ===============================================================
def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if (
        not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt"))
        and module_name == "pose_projector"
    ):
        module_name = "proprio_projector"

    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: "InferenceConfig",
    device: torch.device,
    module_args: dict,
    to_bf16: bool = False,
):
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(
            module_name,
            cfg.vla_path,
            cfg.resume_step,
            device=str(device),
        )
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)

    return module.to(device)


class InferenceConfig:
    resume: bool = True
    vla_path: str = "/nas/sujinkim/model/goto/sim/20260323_224/run#2/omnivla-original-balance--400000_chkpt/"
    resume_step: Optional[int] = 400000
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0


def define_model(cfg: InferenceConfig):
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Loading OpenVLA Model `{cfg.vla_path}`")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPOSE_DIM: {POSE_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.to(dtype=torch.bfloat16, device=device)

    pose_projector = init_module(
        ProprioProjector,
        "pose_projector",
        cfg,
        device,
        {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM},
    )

    action_head = init_module(
        L1RegressionActionHead_idcat,
        "action_head",
        cfg,
        device,
        {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM},
        to_bf16=True,
    )

    num_patches = (
        vla.vision_backbone.get_num_patches()
        * vla.vision_backbone.get_num_images_in_input()
    )
    num_patches += 1  # for goal pose

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    return vla, action_head, pose_projector, device, num_patches, action_tokenizer, processor


# ===============================================================
# OmniVLA client
# ===============================================================
class OmniVLAClient:
    def __init__(self, goal: str = "marker1", save_dir: str = "./results", tick_rate: float = 3.0):
        if goal not in goal_poses:
            raise ValueError(f"Unknown goal: {goal}")

        self.goal = goal
        self.goal_pose = goal_poses[goal]
        self.goal_image_PIL = Image.open(goal_image_paths[goal]).convert("RGB")
        self.lan_inst_prompt = (
            "No language instruction"
            if not lan_prompt
            else f"What action should the robot take to go to {goal}?"
        )
        self.metric_waypoint_spacing = WAYPOINT_SPACING
        self.tick_rate = tick_rate
        self.count_id = 0

        self.base_save_dir = save_dir
        os.makedirs(self.base_save_dir, exist_ok=True)

        self.datastore_path_image = None

        cfg = InferenceConfig()
        (
            self.vla,
            self.action_head,
            self.pose_projector,
            self.device,
            self.num_patches,
            self.action_tokenizer,
            self.processor,
        ) = define_model(cfg)

        self.vla = self.vla.eval()
        self.action_head = self.action_head.eval()
        self.pose_projector = self.pose_projector.eval()

    def get_next_episode_index(self) -> int:
        existing = []
        for name in os.listdir(self.base_save_dir):
            full_path = os.path.join(self.base_save_dir, name)
            if os.path.isdir(full_path) and name.isdigit():
                existing.append(int(name))

        if not existing:
            return 0
        return max(existing) + 1

    def set_episode_save_dir(self, episode_idx: int):
        self.datastore_path_image = os.path.join(self.base_save_dir, f"{episode_idx:03d}")
        os.makedirs(self.datastore_path_image, exist_ok=True)
        self.count_id = 0
        print(f"[SAVE DIR] {self.datastore_path_image}")

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return math.atan2(math.sin(theta), math.cos(theta))

    def _world_to_relative_pose(
        self,
        robot_pose_world: Tuple[float, float, float],
        goal_pose_world: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, float]:
        xr, yr, yaw_r = robot_pose_world
        xg, yg, yaw_g = goal_pose_world

        dx = xg - xr
        dy = yg - yr

        x_rel = math.cos(yaw_r) * dx + math.sin(yaw_r) * dy
        y_rel = -math.sin(yaw_r) * dx + math.cos(yaw_r) * dy
        dyaw = self._wrap_angle(yaw_g - yaw_r)

        goal_distance = math.sqrt(x_rel**2 + y_rel**2)

        x_rel_norm = x_rel / self.metric_waypoint_spacing
        y_rel_norm = y_rel / self.metric_waypoint_spacing

        goal_pose_cos_sin = np.array(
            [x_rel_norm, y_rel_norm, math.cos(dyaw), math.sin(dyaw)],
            dtype=np.float32,
        )
        return goal_pose_cos_sin, goal_distance

    @staticmethod
    def decode_image_b64(image_b64: str) -> Image.Image:
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    # ===========================================================
    # Dataset formatting
    # ===========================================================
    def collator_custom(self, instances, model_max_length, pad_token_id):
        IGNORE_INDEX = -100

        input_ids = pad_sequence(
            [inst["input_ids"] for inst in instances],
            batch_first=True,
            padding_value=pad_token_id,
        )
        labels = pad_sequence(
            [inst["labels"] for inst in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        input_ids, labels = (
            input_ids[:, :model_max_length],
            labels[:, :model_max_length],
        )
        attention_mask = input_ids.ne(pad_token_id)

        pixel_values = [inst["pixel_values_current"] for inst in instances]
        pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
        pixel_values = torch.cat(
            (torch.stack(pixel_values), torch.stack(pixel_values_goal)),
            dim=1,
        )

        actions = torch.stack(
            [torch.from_numpy(inst["actions"].copy()) for inst in instances]
        )
        goal_pose = torch.stack(
            [torch.from_numpy(inst["goal_pose"].copy()) for inst in instances]
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "actions": actions,
            "goal_pose": goal_pose,
        }

    def transform_datatype(
        self,
        inst_obj,
        actions,
        goal_pose_cos_sin,
        current_image_PIL,
        goal_image_PIL,
        prompt_builder,
        action_tokenizer,
        base_tokenizer,
        image_transform,
        predict_stop_token=True,
    ):
        IGNORE_INDEX = -100

        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = "".join(action_tokenizer(future_actions))
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
                {
                    "from": "human",
                    "value": f"What action should the robot take to {inst_obj}?",
                },
                {"from": "gpt", "value": action_chunk_string},
            ]

        pb = prompt_builder("openvla")
        for turn in conversation:
            pb.add_turn(turn["from"], turn["value"])

        input_ids = torch.tensor(
            base_tokenizer(pb.get_prompt(), add_special_tokens=True).input_ids
        )
        labels = input_ids.clone()
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX

        pixel_values_current = image_transform(current_image_PIL)
        pixel_values_goal = image_transform(goal_image_PIL)

        return {
            "pixel_values_current": pixel_values_current,
            "pixel_values_goal": pixel_values_goal,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": "lelan",
            "actions": actions.astype(np.float32),
            "goal_pose": goal_pose_cos_sin.astype(np.float32),
        }

    def data_transformer_omnivla(self, current_image_PIL, goal_pose_loc_norm):
        # dummy action chunk only to preserve training tensor layout
        actions = np.random.rand(8, 4).astype(np.float32)

        batch_data = self.transform_datatype(
            self.lan_inst_prompt,
            actions,
            goal_pose_loc_norm,
            current_image_PIL,
            self.goal_image_PIL,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            base_tokenizer=self.processor.tokenizer,
            image_transform=self.processor.image_processor.apply_transform,
        )

        return self.collator_custom(
            [batch_data],
            self.processor.tokenizer.model_max_length,
            self.processor.tokenizer.pad_token_id,
        )

    # ===========================================================
    # Forward pass
    # ===========================================================
    def run_forward_pass(self, batch):
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
        else:
            raise RuntimeError("Unsupported modality combination")

        autocast_enabled = self.device.type == "cuda"

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output: CausalLMOutputWithPast = self.vla(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(self.device),
                modality_id=modality_id.to(torch.bfloat16).to(self.device),
                labels=batch["labels"].to(self.device),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(self.device),
                proprio_projector=self.pose_projector,
                use_film=False,
            )

        ground_truth_token_ids = batch["labels"][:, 1:].to(self.device)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, self.num_patches : -1]
        batch_size = batch["input_ids"].shape[0]

        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )

        with torch.no_grad():
            predicted_actions = self.action_head.predict_action(
                actions_hidden_states,
                modality_id.to(torch.bfloat16).to(self.device),
            )

        return predicted_actions, modality_id

    # ===========================================================
    # Policy
    # ===========================================================
    def compute_cmd_vel_from_waypoint(self, dx: float, dy: float, hx: float, hy: float):
        EPS = 1e-8
        DT = 1 / self.tick_rate

        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_vel_value = 0
            angular_vel_value = clip_angle(np.arctan2(hy, hx)) / DT

        elif np.abs(dx) < EPS:
            linear_vel_value = 0
            angular_vel_value = np.sign(dy) * np.pi / (2 * DT)

        else:
            linear_vel_value = dx / DT
            angular_vel_value = np.arctan(dy / dx) / DT

        # allow backward
        linear_vel_value = np.clip(linear_vel_value, -1.0, 1.0)
        angular_vel_value = np.clip(angular_vel_value, -1.5, 1.5)

        # velocity limitation
        maxv, maxw = 0.8, 0.7

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

        return float(linear_vel_value_limit), float(angular_vel_value_limit)
    
        
    def predict_action(self, pose_dict: dict, image_b64: str):
        robot_pose_world = (
            float(pose_dict["x"]),
            float(pose_dict["y"]),
            float(pose_dict["yaw"]),
        )

        goal_pose_loc_norm, goal_distance = self._world_to_relative_pose(
            robot_pose_world,
            self.goal_pose,
        )

        current_image_PIL = self.decode_image_b64(image_b64)

        batch = self.data_transformer_omnivla(
            current_image_PIL=current_image_PIL,
            goal_pose_loc_norm=goal_pose_loc_norm,
        )

        actions, modality_id = self.run_forward_pass(batch)
        waypoints = actions.float().cpu().numpy()

        waypoint_select = 4
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= self.metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint

        cmd_vel_v, cmd_vel_w = self.compute_cmd_vel_from_waypoint(dx, dy, hx, hy)
                        
        vis_payload = {
            "current_image_PIL": current_image_PIL,
            "goal_img": self.goal_image_PIL,
            "goal_pose": goal_pose_loc_norm,
            "waypoints": waypoints[0],
            "linear_vel": float(cmd_vel_v),
            "angular_vel": float(cmd_vel_w),
            "metric_waypoint_spacing": self.metric_waypoint_spacing,
            "mask_number": modality_id.cpu().numpy(),
        }

        return (
            float(cmd_vel_v),
            float(cmd_vel_w),
            float(goal_distance),
            vis_payload,
        )

    # ===========================================================
    # Visualization
    # ===========================================================
    def save_robot_behavior(
        self,
        current_image_PIL,
        goal_img,
        goal_pose,
        waypoints,
        linear_vel,
        angular_vel,
        metric_waypoint_spacing,
        mask_number,
    ):
        fig = plt.figure(figsize=(16, 8), dpi=100)
        gs = fig.add_gridspec(2, 2)
        ax_ob = fig.add_subplot(gs[0, 0])
        ax_goal = fig.add_subplot(gs[1, 0])
        ax_graph_pos = fig.add_subplot(gs[:, 1])

        ax_ob.imshow(np.array(current_image_PIL).astype(np.uint8))
        ax_goal.imshow(np.array(goal_img).astype(np.uint8))

        x_seq = waypoints[:, 0]
        y_seq_inv = -waypoints[:, 1]
        ax_graph_pos.plot(
            np.insert(y_seq_inv, 0, 0.0),
            np.insert(x_seq, 0, 0.0),
            linewidth=2.0,
            markersize=6,
            marker="o",
        )

        mask_type = int(mask_number[0])
        mask_texts = [
            "satellite only",
            "pose and satellite",
            "satellite and image",
            "all",
            "pose only",
            "pose and image",
            "image only",
            "language only",
            "language and pose",
        ]
        if mask_type < len(mask_texts):
            ax_graph_pos.annotate(
                mask_texts[mask_type],
                xy=(1.0, 0.0),
                xytext=(-20, 20),
                fontsize=12,
                textcoords="offset points",
            )

        ax_ob.set_title("Current image")
        ax_goal.set_title("Goal image")

        if mask_type in [1, 3, 4, 5, 8]:
            ax_graph_pos.plot(-goal_pose[1], goal_pose[0], marker="*", markersize=12)

        ax_graph_pos.set_xlim(-3.0, 3.0)
        ax_graph_pos.set_ylim(-0.1, 10.0)
        ax_graph_pos.set_title(
            f"Predicted trajectory | v={linear_vel:.3f}, w={angular_vel:.3f}"
        )

        save_path = os.path.join(
            self.datastore_path_image,
            f"{self.count_id:05d}_ex.jpg",
        )
        self.count_id += 1
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)


# ===============================================================
# Main
# ===============================================================
def main():
    ISAACSIM_HOST = "192.168.0.180"
    SIM_PORT = 8765
    CMD_PORT = 8766

    HOLD_SECONDS_BEFORE_RESET = 3.0
    HOLD_CMD_DT = 0.1

    STOP_LINEAR = 0.02
    STOP_ANGULAR = 0.02
    STOP_COUNT_THRESH = 15

    LOOP_SLEEP_DT = 0.005

    INFER_HZ = 3.0
    SAVE_FPS = 30.0

    INFER_DT = 1.0 / INFER_HZ      # 0.2 sec
    SAVE_DT = 1.0 / SAVE_FPS       # 0.0333 sec

    sim = JsonSocketClient(ISAACSIM_HOST, SIM_PORT)
    cmd_sender = JsonLineSender(ISAACSIM_HOST, CMD_PORT)
    cli = OmniVLAClient(goal="marker3", save_dir="./results", tick_rate=INFER_HZ)

    episode_idx = cli.get_next_episode_index()
    global_step = 0
    episode_step = 0
    stop_counter = 0

    last_infer_time = 0.0
    last_save_time = 0.0

    latest_linear = 0.0
    latest_angular = 0.0
    latest_goal_distance = float("inf")
    latest_vis_payload = None

    def hold_still(duration_sec: float):
        hold_start = time.time()
        while time.time() - hold_start < duration_sec:
            cmd_sender.send({"linear": 0.0, "angular": 0.0})
            time.sleep(HOLD_CMD_DT)

    def start_new_episode(ep_idx: int):
        nonlocal last_infer_time, last_save_time
        nonlocal latest_linear, latest_angular, latest_goal_distance, latest_vis_payload

        cli.set_episode_save_dir(ep_idx)
        reset_resp = sim.request({"cmd": "reset"})
        print(f"[SIM RESET][EP {ep_idx:03d}] {reset_resp}")

        last_infer_time = 0.0
        last_save_time = 0.0
        latest_linear = 0.0
        latest_angular = 0.0
        latest_goal_distance = float("inf")
        latest_vis_payload = None

        return reset_resp

    try:
        ping = sim.request({"cmd": "ping"})
        print("[SIM PING]", ping)

        start_new_episode(episode_idx)

        while True:
            now = time.time()

            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip()

                if key == "r":
                    print("[MANUAL RESET]")

                    hold_still(1.0)

                    episode_idx += 1
                    episode_step = 0
                    stop_counter = 0

                    start_new_episode(episode_idx)
                    continue

            obs = sim.request({"cmd": "get_obs"})
            if not obs.get("ok", False):
                raise RuntimeError(obs)

            # -----------------------------
            # 5 Hz inference
            # -----------------------------
            if now - last_infer_time >= INFER_DT:
                (
                    latest_linear,
                    latest_angular,
                    latest_goal_distance,
                    latest_vis_payload,
                ) = cli.predict_action(
                    obs["pose"],
                    obs["image_b64"],
                )

                cmd_sender.send(
                    {"linear": latest_linear, "angular": latest_angular}
                )

                is_stop_cmd = (
                    abs(latest_linear) < STOP_LINEAR and abs(latest_angular) < STOP_ANGULAR
                )
                if is_stop_cmd:
                    stop_counter += 1
                else:
                    stop_counter = 0

                print(
                    f"[EP {episode_idx:03d} | EP_STEP {episode_step:05d} | STEP {global_step:07d}] "
                    f"v={latest_linear:.3f}, w={latest_angular:.3f}, "
                    f"goal_dist={latest_goal_distance:.3f}, stop_count={stop_counter}"
                )

                episode_step += 1
                global_step += 1
                last_infer_time = now

            # -----------------------------
            # 30 FPS save
            # -----------------------------
            if latest_vis_payload is not None and (now - last_save_time >= SAVE_DT):
                cli.save_robot_behavior(**latest_vis_payload)
                last_save_time = now

            if stop_counter >= STOP_COUNT_THRESH:
                print(
                    f"[DONE][EP {episode_idx:03d}] "
                    f"STOP command detected for {STOP_COUNT_THRESH} consecutive steps. "
                    f"Holding still for {HOLD_SECONDS_BEFORE_RESET:.1f}s before reset..."
                )

                hold_still(HOLD_SECONDS_BEFORE_RESET)

                episode_idx += 1
                episode_step = 0
                stop_counter = 0

                start_new_episode(episode_idx)
                continue

            time.sleep(LOOP_SLEEP_DT)

    finally:
        try:
            hold_still(0.3)
        except Exception:
            pass

        cmd_sender.close()
        sim.close()

if __name__ == "__main__":
    main()