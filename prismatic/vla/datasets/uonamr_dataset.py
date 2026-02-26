import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Type, Optional, List, Union

from torchvision.transforms.functional import to_pil_image, resize, to_tensor
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

from prismatic.vla.constants import IGNORE_INDEX
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform

# Repo 기준 modality id (train_omnivla.py 내부 마스킹/통계가 0~8 기반이라 9는 비추)
IMAGE_ONLY = 6
LANGUAGE_ONLY = 7
LANGUAGE_AND_POSE = 8


class UONAMR_Dataset(LeRobotDataset):
    """
    OmniVLA (NHirose/OmniVLA) train_omnivla.py + PaddedCollatorForActionPrediction_Nav_MMN
    파이프라인에 맞춘 최종 호환 Dataset (LeRobot 기반).

    반환 dict (중요 키/타입):
      - pixel_values: torch.FloatTensor
      - pixel_values_goal: torch.FloatTensor
      - input_ids: torch.LongTensor (1D)
      - labels: torch.LongTensor (1D)
      - actions: np.ndarray float32, (H,4)
      - action_select_mask: np.ndarray float32 scalar
      - goal_pose: np.ndarray float32, (4,)
      - obj_pose_norm: np.ndarray float32, (2,)
      - cur_image: np.ndarray float32, (3*(context_size+1), 96, 96)
      - goal_image_8: np.ndarray float32, (3, 96, 96)
      - temp_dist: np.ndarray float32 scalar
      - modality_id: int
      - img_PIL, gimg_PIL: PIL.Image (VISUALIZE 안전)
      - dataset_name: str
      - lan_prompt: str (디버그/로깅용)
    """

    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        root: Path,
        modality: int = LANGUAGE_ONLY,  # 안전 default
        predict_stop_token: bool = True,
        action_horizon: int = 8,
        action_spacing: int = 1,
        context_size: int = 5,
        context_spacing: int = 1,
        dataset_framerate: int = 15,   # 너 조건: episode마다 15fps 고정
        len_traj_pred: int = 8,        # action horizon과 동일하게 권장
        max_v: float = 0.5,
        instruction_prefix: str = "Instruction: ",
        default_instruction: str = "Navigate to the goal.",
        mbra_image_hw: int = 96,
        goal_as_last_frame: bool = True,
    ):
        self.dt = 1.0 / float(dataset_framerate)
        self.action_spacing = int(action_spacing)
        self.action_horizon = int(action_horizon)
        self.context_size = int(context_size)
        self.context_spacing = int(context_spacing)
        self.modality = int(modality)

        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.predict_stop_token = bool(predict_stop_token)

        self.len_traj_pred = int(len_traj_pred)
        self.max_v = float(max_v)

        self.instruction_prefix = instruction_prefix
        self.default_instruction = default_instruction

        self.mbra_image_hw = int(mbra_image_hw)
        self.goal_as_last_frame = bool(goal_as_last_frame)

        super().__init__(
            repo_id="uon_amr",
            download_videos=True,
            root=root,
            delta_timestamps={
                "observation.images.front_rgb": [
                    i * self.context_spacing * self.dt for i in range(-self.context_size, 1)
                ],
                "action": [i * self.action_spacing * self.dt for i in range(self.action_horizon)],
            },
        )

    # --------------------------
    # Action: (v,w) -> (x,y,cos,sin)
    # --------------------------
    def integrate_velocity(self, actions_vw: torch.Tensor) -> torch.Tensor:
        """
        actions_vw: (H,2) torch
        returns:    (H,4) torch float32
        """
        H = actions_vw.shape[0]
        positions = torch.zeros((H + 1, 2), dtype=torch.float32)
        headings = torch.zeros(H + 1, dtype=torch.float32)

        for i in range(1, H + 1):
            v = actions_vw[i - 1, 0]
            w = actions_vw[i - 1, 1]
            direction = torch.tensor(
                [torch.cos(headings[i - 1]), torch.sin(headings[i - 1])],
                dtype=torch.float32,
            )
            positions[i] = positions[i - 1] + v * direction * self.dt
            headings[i] = headings[i - 1] + w * self.dt

        future_pos = positions[1:]
        future_headings = headings[1:]

        return torch.stack(
            [
                future_pos[:, 0] / self.max_v,
                future_pos[:, 1] / self.max_v,
                torch.cos(future_headings),
                torch.sin(future_headings),
            ],
            dim=-1,
        ).to(torch.float32)

    # --------------------------
    # Frame conversion helpers
    # --------------------------
    @staticmethod
    def _frame_to_pil(frame: Any):
        if isinstance(frame, torch.Tensor):
            x = frame
            if x.ndim == 3 and x.shape[0] in (1, 3):          # (C,H,W)
                pass
            elif x.ndim == 3 and x.shape[-1] in (1, 3):       # (H,W,C)
                x = x.permute(2, 0, 1)
            else:
                raise ValueError(f"Unexpected frame tensor shape: {tuple(x.shape)}")
            return to_pil_image(x.detach().cpu()).convert("RGB")

        if isinstance(frame, np.ndarray):
            x = frame
            if x.ndim == 3 and x.shape[-1] in (1, 3):         # (H,W,C)
                pass
            elif x.ndim == 3 and x.shape[0] in (1, 3):        # (C,H,W)
                x = np.transpose(x, (1, 2, 0))
            else:
                raise ValueError(f"Unexpected frame np shape: {x.shape}")
            return to_pil_image(x).convert("RGB")

        raise TypeError(f"Unsupported frame type: {type(frame)}")

    def _make_mbra_images(
        self,
        history_frames: Any,
        goal_pil,
    ) -> Dict[str, np.ndarray]:
        """
        cur_image:    (3*(context_size+1), 96, 96) float32
        goal_image_8: (3, 96, 96) float32
        """
        hw = self.mbra_image_hw

        # history_frames -> list of frames
        if isinstance(history_frames, torch.Tensor):
            frames_list = [history_frames[i] for i in range(history_frames.shape[0])]
        elif isinstance(history_frames, (list, tuple)):
            frames_list = list(history_frames)
        elif isinstance(history_frames, np.ndarray):
            frames_list = [history_frames[i] for i in range(history_frames.shape[0])]
        else:
            frames_list = [history_frames]

        needed = self.context_size + 1
        if len(frames_list) >= needed:
            frames_list = frames_list[-needed:]
        else:
            last = frames_list[-1]
            frames_list = [last] * (needed - len(frames_list)) + frames_list

        t_list = []
        for fr in frames_list:
            pil = self._frame_to_pil(fr)
            pil = resize(pil, [hw, hw])
            t_list.append(to_tensor(pil).to(torch.float32))   # (3,hw,hw)

        cur_image = torch.cat(t_list, dim=0)                  # (3*(ctx+1),hw,hw)

        goal_rs = resize(goal_pil, [hw, hw])
        goal_image = to_tensor(goal_rs).to(torch.float32)     # (3,hw,hw)

        return {
            "cur_image": cur_image.numpy().astype(np.float32),
            "goal_image_8": goal_image.numpy().astype(np.float32),
        }

    # --------------------------
    # Main getitem
    # --------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        ep_id = int(item["episode_index"].item())
        ep_info = self.meta.episodes[ep_id]

        # ---- Language prompt ----
        lan_prompt = self.default_instruction
        try:
            task_idx = int(item["task_index"].item())
            lan_prompt = str(self.meta.tasks.index[task_idx])
        except Exception:
            lan_prompt = self.default_instruction

        # ---- Images: current + goal ----
        vid_key = "observation.images.front_rgb"

        # current image (history last frame)
        img_pil = self._frame_to_pil(item[vid_key][-1])

        # goal image: episode 마지막 프레임 사용 (너 로직 유지)
        video_path = self.root / self.meta.get_video_file_path(ep_id, vid_key)

        if self.goal_as_last_frame:
            goal_ts = self.hf_dataset[ep_info["dataset_to_index"] - 1]["timestamp"].item()
        else:
            # fallback: 현재 timestamp를 goal로 (원하면 바꿔)
            goal_ts = item["timestamp"].item() if "timestamp" in item else 0.0

        from_ts = float(ep_info.get(f"videos/{vid_key}/from_timestamp", 0.0))
        goal_raw = decode_video_frames(
            video_path, [from_ts + float(goal_ts)], tolerance_s=0.0001
        ).squeeze(0)

        if isinstance(goal_raw, np.ndarray) and goal_raw.ndim == 4:
            goal_frame = goal_raw[0]
        else:
            goal_frame = goal_raw

        gimg_pil = self._frame_to_pil(goal_frame)

        # OmniVLA vision encoder inputs (torch tensors)
        pixel_values = self.image_transform(img_pil)
        pixel_values_goal = self.image_transform(gimg_pil)

        # MBRA inputs (numpy)
        mbra = self._make_mbra_images(item[vid_key], gimg_pil)

        # ---- Actions ----
        actions_vw = item["action"]
        if not isinstance(actions_vw, torch.Tensor):
            actions_vw = torch.tensor(actions_vw)

        # horizon 방어 + 패딩
        H_raw = int(actions_vw.shape[0])
        H = min(H_raw, self.len_traj_pred)
        actions_4d = self.integrate_velocity(actions_vw[:H])

        if H < self.len_traj_pred:
            pad = actions_4d[-1:].repeat(self.len_traj_pred - H, 1)
            actions_4d = torch.cat([actions_4d, pad], dim=0)

        # numpy actions (collator 호환)
        actions_np = actions_4d.detach().cpu().numpy().astype(np.float32)  # (len_traj_pred,4)

        # ---- ActionTokenizer: Frodobots style ----
        current_action = actions_4d[0]
        future_actions = actions_4d[1:]
        current_action_string = self.action_tokenizer(current_action)
        future_actions_string = "".join(self.action_tokenizer(future_actions))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        # ---- Prompt: OpenVLA style ----
        human_query = f"{self.instruction_prefix}{lan_prompt} Reach the destination shown in the goal image."

        conversation = [
            {"from": "human", "value": human_query},
            {"from": "gpt", "value": action_chunk_string},
        ]

        prompt_builder = self.prompt_builder_fn("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids_list = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        labels = input_ids.clone()

        # Frodobots 방식: action chunk + stop 토큰만 loss 계산
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # ---- Other required fields (numpy) ----
        # goal_pose / obj_pose_norm은 일부 loss 경로에서 참조됨 -> 0이라도 넣어두기
        goal_pose_np = np.zeros((4,), dtype=np.float32)
        obj_pose_norm_np = goal_pose_np[:2].copy().astype(np.float32)

        # temp_dist는 코드에서 clip만 함
        temp_dist_np = np.array(10.0, dtype=np.float32)

        # action_select_mask: raw action loss 쓰려면 1.0
        action_select_mask_np = np.array(1.0, dtype=np.float32)

        # modality_id: repo에서 0~8 범위 전제를 깔고 있는 로직이 있어 안전하게 7 고정 추천
        modality_id = int(self.modality)
        if modality_id not in (IMAGE_ONLY, LANGUAGE_ONLY, LANGUAGE_AND_POSE):
            modality_id = LANGUAGE_ONLY

        return dict(
            dataset_name="uon_amr",
            modality_id=modality_id,  # ✅ int

            pixel_values=pixel_values,           # ✅ torch
            pixel_values_goal=pixel_values_goal, # ✅ torch

            input_ids=input_ids,                 # ✅ torch
            labels=labels,                       # ✅ torch

            actions=actions_np,                  # ✅ numpy
            action_select_mask=action_select_mask_np,

            goal_pose=goal_pose_np,
            obj_pose_norm=obj_pose_norm_np,
            temp_dist=temp_dist_np,

            cur_image=mbra["cur_image"],         # ✅ numpy (3*(ctx+1),96,96)
            goal_image_8=mbra["goal_image_8"],   # ✅ numpy (3,96,96)

            img_PIL=img_pil,
            gimg_PIL=gimg_pil,

            lan_prompt=lan_prompt,
        )