import torch
import random

import numpy as np
from pathlib import Path
from typing import Dict, Any, Type
from PIL import Image

from torchvision.transforms.functional import to_pil_image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import einops

from prismatic.vla.constants import IGNORE_INDEX
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform


def trans_mat(
    pos: float | np.ndarray | torch.Tensor, yaw: float | np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """Return homogeneous transform matrix for position and yaw."""
    if isinstance(yaw, torch.Tensor):
        return torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), pos[0]],
                [torch.sin(yaw), torch.cos(yaw), pos[1]],
                [torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)],
            ]
        )
    else:
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), pos[0]],
                [np.sin(yaw), np.cos(yaw), pos[1]],
                [0.0, 0.0, 1.0],
            ]
        )

def to_local_coords_yaw(
    positions: np.ndarray | torch.Tensor,
    curr_pos: np.ndarray | torch.Tensor,
    curr_yaw: float | np.ndarray | torch.Tensor,
    goal_yaw: float | np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    Return relative transform matrix between current frame (curr_pos, curr_yaw)
    and goal frame defined by positions[0] and goal_yaw.
    """
    cur_mat = trans_mat(curr_pos, curr_yaw)
    goal_mat = trans_mat(positions[0], goal_yaw)
    cur_mat_inv = torch.linalg.inv(cur_mat) if isinstance(cur_mat, torch.Tensor) else np.linalg.inv(cur_mat)
    return cur_mat_inv @ goal_mat


class UONAMR_Dataset(LeRobotDataset):
    """
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
        predict_stop_token: bool = True,
        metric_spacing: float = 0.1,
        action_horizon: int = 8, # = waypoint spacing
        action_spacing: int = 3, # 15 framerate 기준 0.2초 간격
        context_size: int = 5,
        context_spacing: int = 3, # 15 framerate 기준 0.2초 간격
        dataset_framerate: int = 15,  
        len_traj_pred: int = 8,       
    ):
        self.dt = 1.0 / float(dataset_framerate)

        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.predict_stop_token = bool(predict_stop_token)
        
        self.metric_spacing = float(metric_spacing)
        self.action_horizon = int(action_horizon)
        self.action_spacing = int(action_spacing)
        self.context_size = int(context_size)
        self.context_spacing = int(context_spacing)
        self.len_traj_pred = int(len_traj_pred)

        super().__init__(
            repo_id="uon_amr",
            download_videos=False,
            root=root,
        )      
        
        # # dataset cache stored as zarr array -> keep as numpy arrays for speed
        # self.dataset_cache = zarr.load(Path(root) / "uon_dataset" / "dataset_cache.zarr")
        # self.dataset_cache = {k: np.asarray(v) for k, v in self.dataset_cache.items()}

        # self.global_positions = np.zeros((len(self.dataset_cache["linear_velocity"]), 2))
        # self.global_headings = np.zeros(len(self.dataset_cache["angular_velocity"]))

        # # 모든 에피소드를 순회하며 전역 좌표 적분
        # for ep_id in range(self.num_episodes):
        #     start = self.episode_data_index["from"][ep_id]
        #     end = self.episode_data_index["to"][ep_id]
            
        #     v_ep = self.dataset_cache["linear_velocity"][start:end]
        #     w_ep = self.dataset_cache["angular_velocity"][start:end]
            
        #     curr_x, curr_y, curr_theta = 0.0, 0.0, 0.0
        #     for i, (v, w) in enumerate(zip(v_ep, w_ep)):
        #         # 전역 좌표 저장
        #         self.global_positions[start + i] = [curr_x, curr_y]
        #         self.global_headings[start + i] = curr_theta
                
        #         # 다음 스텝 적분
        #         curr_x += v * np.cos(curr_theta) * self.dt
        #         curr_y += v * np.sin(curr_theta) * self.dt
        #         curr_theta += w * self.dt
                
    # # --------------------------
    # # Action: (v,w) -> (x,y,cos,sin)
    # # --------------------------
    # def integrate_velocity(self, actions_vw: torch.Tensor) -> torch.Tensor:
    #     """
    #     actions_vw: (H,2) torch
    #     returns:    (H,4) torch float32
    #     """
    #     H = actions_vw.shape[0]
    #     positions = torch.zeros((H + 1, 2), dtype=torch.float32)
    #     headings = torch.zeros(H + 1, dtype=torch.float32)

    #     for i in range(1, H + 1):
    #         v = actions_vw[i - 1, 0]
    #         w = actions_vw[i - 1, 1]
    #         direction = torch.tensor(
    #             [torch.cos(headings[i - 1]), torch.sin(headings[i - 1])],
    #             dtype=torch.float32,
    #         )
    #         positions[i] = positions[i - 1] + v * direction * self.dt
    #         headings[i] = headings[i - 1] + w * self.dt

    #     future_pos = positions[1:]
    #     future_headings = headings[1:]

    #     return torch.stack(
    #         [
    #             future_pos[:, 0] / self.metric_waypoint_spacing, # Normalize  
    #             future_pos[:, 1] / self.metric_waypoint_spacing, # Normalize
    #             torch.cos(future_headings), # Trigonometry 적용 
    #             torch.sin(future_headings), # Trigonometry 적용
    #         ],
    #         dim=-1,
    #     ).to(torch.float32)

    # --------------------------
    # Main getitem
    # --------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        ep_id = int(item["episode_index"].item()) # 현재 idx 가 속한 episode id        
        episode_metadata = self.meta.episodes[ep_id]
        ep_start_idx = episode_metadata["dataset_from_index"]
        ep_end_idx = episode_metadata["dataset_to_index"]
        
        # Context/history 만들기
        context_indices = []
        for i in range(-self.context_size, 1):
            t_idx = idx + (i * self.context_spacing)
            # 에피소드 시작점보다 작아지면 시작점 프레임으로 패딩 
            context_indices.append(max(t_idx, ep_start_idx))
            
        context_images = []
        for c_idx in context_indices:
            c_item = super().__getitem__(c_idx)
            context_images.append(c_item["observation.images.front_rgb"]) # (3,H,W)
            
        context_stack = torch.stack(context_images)  # (T,3,H,W) 형태로 쌓기
        cur_image_context = einops.rearrange(context_stack, "t c h w -> (t c) h w")  # Flatten channel: (T*3,H,W)
        
        # 현재 이미지 = context의 마지막 프레임
        current_img_tensor = context_images[-1]

        # 에피소드의 마지막 프레임을 goal image 로 사용
        last_frame_idx = ep_end_idx - 1
        goal_img_tensor = super().__getitem__(last_frame_idx)["observation.images.front_rgb"]
        
        # 전처리
        pil_current = to_pil_image(current_img_tensor)
        pil_goal = to_pil_image(goal_img_tensor)
        
        pixel_values = self.image_transform(pil_current)
        pixel_values_goal = self.image_transform(pil_goal)
        
        # ---------- Action ----------
        # 미래 v, w 데이터 로드
        total_steps_needed = self.action_spacing * self.action_horizon
        actual_end_idx = int(min(idx + total_steps_needed, ep_end_idx))
        future_actions_list = self.hf_dataset[idx : actual_end_idx]["action"] 

        # 리스트 안의 요소가 텐서라면 stack을, 아니라면 as_tensor를 사용해야 함
        if isinstance(future_actions_list[0], torch.Tensor):
            future_actions_tensor = torch.stack(future_actions_list)
        else:
            future_actions_tensor = torch.as_tensor(future_actions_list)

        v_seq = future_actions_tensor[:, 0] 
        w_seq = future_actions_tensor[:, 1]

        # 에피소드 끝 Padding (마지막 속도 유지)
        if v_seq.shape[0] < total_steps_needed:
            pad_len = total_steps_needed - v_seq.shape[0]
            v_pad = v_seq[-1].repeat(pad_len)
            w_pad = w_seq[-1].repeat(pad_len)
            v_seq = torch.cat([v_seq, v_pad])
            w_seq = torch.cat([w_seq, w_pad])
            
        # 모은 데이터는 w > 0 일 떄 우회전 --> 표준으로 변환필요: w > 0 일 때 좌회전이 되도록 (y좌표가 좌측+이므로)
        # !! 로봇 제어 시 계산한 w 값에 -1 곱해서 제어해야 함 !!
        w_seq = -w_seq
        
        # Integrate velocity: (v,w) -> (x,y,cos,sin)
        curr_x, curr_y, curr_theta = 0.0, 0.0, 0.0
        action_waypoints = []
        for i in range(total_steps_needed):
            v, w = v_seq[i].item(), w_seq[i].item()
            curr_x += v * np.cos(curr_theta) * self.dt
            curr_y += v * np.sin(curr_theta) * self.dt
            curr_theta += w * self.dt
            
            if (i + 1) % self.action_spacing == 0:
                action_waypoints.append([
                    curr_x / self.metric_spacing,
                    curr_y / self.metric_spacing,
                    np.cos(curr_theta),
                    np.sin(curr_theta)
                ])
                
        actions = torch.tensor(action_waypoints, dtype=torch.float32)
        
        # Data augmentation
        if random.random() > 0.5:
            # 모델 입력용 이미지 텐서
            pixel_values = self.image_transform(pil_current.transpose(Image.FLIP_LEFT_RIGHT))
            pixel_values_goal = self.image_transform(pil_goal.transpose(Image.FLIP_LEFT_RIGHT))
            cur_image_context = torch.flip(cur_image_context, [2])
            # 액션 trajectory 반전
            actions[:, 1] *= -1 # dy 반전
            actions[:, 3] *= -1 # sin(yaw) 반전
            # 디버깅/로깅용 PIL 이미지
            pil_current = pil_current.transpose(Image.FLIP_LEFT_RIGHT)
            pil_goal = pil_goal.transpose(Image.FLIP_LEFT_RIGHT)
        
        # action tokenizer 적용
        current_action = actions[0]
        future_actions = actions[1:]
        current_action_string = self.action_tokenizer(current_action)
        future_actions_string = "".join(self.action_tokenizer(future_actions))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)
        
        # 1.0: raw action, 0.0: MBRA synthetic action
        action_select_mask = torch.tensor(1.0)    
        
        # # ---------- Goal pose (최종 목적지 상대 좌표 계산) ----------
        # # 현재와 마지막 지점의 미리 계산된 전역 좌표 불러오기
        # curr_pos = self.global_positions[idx]
        # curr_heading = self.global_headings[idx]
        # goal_pos = self.global_positions[last_frame_idx]
        # goal_heading = self.global_headings[last_frame_idx]
        
        # # 상대 좌표 계산
        # rel_mat = to_local_coords_yaw(goal_pos[None], curr_pos, curr_heading, goal_heading)
        # # rel_mat[0,2] = 상대 dx,
        # # rel_mat[1,2] = 상대 dy,
        # # rel_mat[1,1] = cos(relative_yaw)
        # # rel_mat[1,0] = sin(relative_yaw)
        # goal_pos_cos_sin = np.array([
        #     rel_mat[0, 2] / self.metric_spacing,  # dx 정규화
        #     rel_mat[1, 2] / self.metric_spacing,  # dy 정규
        #     rel_mat[1, 1],                        # cos(relative_yaw)
        #     rel_mat[1, 0],                        # sin(relative_yaw)
        # ], dtype=np.float32)
        
        # # 목표까지의 거리감
        # # 현재 위치에서 에피소드 끝까지 몇 스텝 남았는지 계산 (wp 단위)
        # distance = (last_frame_idx - idx) // self.action_spacing
        
        # # 상황에 따라 좌표만 보거나 이미지도 같이 보거나 할 수 있도록 modatliy를 유연하게 적용하는 기법
        # # 0:"satellite only", 
        # # 1:"pose and satellite", 
        # # 2:"satellite and image",
        # # 3:"all",
        # # 4:"pose only",
        # # 5:"pose and image", 
        # # 6:"image only", 
        # # 7:"language only", 
        # # 8:"language and pose"        
        # modality_list = [4, 5, 7, 8, 6]   
        # if distance <= 20:
        #     modality_id = random.choice(modality_list)
        # else:
        #     modality_id = random.choice(modality_list[0:4]) #distance is long --> no image only
                
        modality_list = [6, 7] 
        modality_id = random.choice(modality_list)
        
        # ---------- Prompt ----------
        lan_prompt = "XXXX"
        try:
            task_idx = int(item["task_index"].item())
            lan_prompt = str(self.meta.tasks.index[task_idx])
        except Exception:
            lan_prompt = "XXXX"
            
        if modality_id == 7:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lan_prompt}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]   
        
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)     
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)    
        
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!     
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
            
        # pose 데이터는 마스킹처리
        goal_pos_cos_sin = np.zeros(4, dtype=np.float32)
        distance = 0  #거리 정보도 무의미
        
        # 물체 포즈 정규화
        # 원래 로봇팔이 물체를 집을 때 물체의 위치를 알려주기 위함 인데, navigatio에서는 물체가 없으므로 그냥 0으로 채움
        obj_pose_norm = np.zeros(2, dtype=np.float32)
        
        if modality_id == 7: 
            pixel_values_goal = torch.zeros_like(pixel_values)  # dummy image for language-only case

        return dict(
            dataset_name="uon_amr",
            modality_id=modality_id,

            pixel_values=pixel_values,           # 현재 시점의 관측 이미지: DinoV2/SigLIP 입력용
            pixel_values_goal=pixel_values_goal, # 목표 지점의 이미지 (modality_id = 6일 때 사용)

            input_ids=input_ids,                 
            labels=labels,                       

            actions=torch.as_tensor(actions),                 
            action_select_mask=action_select_mask,

            goal_pose=goal_pos_cos_sin,
            obj_pose_norm=obj_pose_norm,
            temp_dist=distance,

            cur_image=cur_image_context,   # context/history: 과거 프레임들을 쌓은 데이터
            goal_image_8=goal_img_tensor,   # Raw Goal Image Tensor

            img_PIL=pil_current, # 디버깅/로그용 PIL 이미지
            gimg_PIL=pil_goal,   # 디버깅/로그용 PIL 이미지 

            lan_prompt=lan_prompt,
        )          
        