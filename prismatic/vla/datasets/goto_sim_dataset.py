import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from prismatic.vla.constants import IGNORE_INDEX


class GotoSim_Dataset(Dataset):
    """
    Raw data structure
    --------------------------
    root/
      action/
        <goal_name>/
            <episode_id>.json
            ...
      rgb/
        <goal_name>/
            <episode_id>/
                000000.png
                000001.png
                ...

    JSON per-episode
    ----------------------------------------------------------
    {
      "episode_id": "0009",
      "goal_name": "goal_marker1",
      "language_instruction": "goal_marker1",
      "goal_pose": {"x": ..., "y": ..., "yaw": ...},
      "trajectory": [
        {
          "index": 0,
          "map_pose": {"x": ..., "y": ..., "yaw": ...},
          "cmd_vel": {...}
        },
        ...
      ]
    }

    Key design choices
    ------------------
    1) stored global poses / waypoints are converted to robot-relative pose/action chunks
    2) goal image can be either
       - a future frame from the same trajectory, or
       - a destination image for the final goal
    3) modality id follows OmniVLA existing convention
       4: pose only
       5: pose + image
       6: image only
       7: language only
       8: language + pose
    4) action labels are waypoint-style [x, y, cos(yaw), sin(yaw)] chunks
    """

    def __init__(
        self,
        root_dir: str,
        image_transform,
        action_tokenizer,
        prompt_builder_fn,
        base_tokenizer,
        image_size: Tuple[int, int] = (96, 96),
        image_size_clip: Tuple[int, int] = (224, 224),
        metric_waypoint_spacing: float = 0.25,
        action_horizon: int = 8,
        action_spacing: int = 5,
        min_future_gap: int = 12, 
        max_future_gap: int = 60,
        predict_stop_token: bool = False, # TODO: ??
        use_flip_aug: bool = False,
        goal_source_probs: Optional[Dict[str, float]] = None,
        modality_probs: Optional[Dict[str, float]] = None,
        allowed_goal_sources: Optional[Dict[str, Sequence[str]]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.image_transform = image_transform
        self.action_tokenizer = action_tokenizer
        self.prompt_builder = prompt_builder_fn
        self.base_tokenizer = base_tokenizer
        self.image_size = image_size
        self.image_size_clip = image_size_clip
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.action_horizon = action_horizon
        self.action_spacing = action_spacing
        self.min_future_gap = min_future_gap
        self.max_future_gap = max_future_gap
        self.predict_stop_token = predict_stop_token
        self.use_flip_aug = use_flip_aug
        
        if goal_source_probs is None:
            goal_source_probs = {
                "future": 0.6,
                "destination": 0.4,
            }

        if modality_probs is None:
            modality_probs = {
                "pose_only": 0.10,       # MOD 4
                "image_pose": 0.45,      # MOD 5
                "image_only": 0.20,      # MOD 6
                "language_only": 0.05,   # MOD 7
                "language_pose": 0.20,   # MOD 8
            }

        if allowed_goal_sources is None:
            allowed_goal_sources = {
                "pose_only": ("future", "destination"),
                "image_pose": ("future", "destination"),
                "image_only": ("future", "destination"),
                "language_only": ("destination",),
                "language_pose": ("destination",),
            }

        assert set(goal_source_probs.keys()) == {"future", "destination"}
        assert abs(sum(goal_source_probs.values()) - 1.0) < 1e-6
        assert set(modality_probs.keys()) == {
            "pose_only", "image_only", "language_only", "image_pose", "language_pose"
        }
        assert abs(sum(modality_probs.values()) - 1.0) < 1e-6
        assert set(allowed_goal_sources.keys()) == set(modality_probs.keys())

        for combo_name, allowed_sources in allowed_goal_sources.items():
            allowed_set = set(allowed_sources)
            assert len(allowed_set) > 0
            assert allowed_set.issubset({"future", "destination"})

        self.goal_source_probs = goal_source_probs
        self.modality_combo_probs = modality_probs
        self.allowed_goal_sources = {
            combo_name: tuple(allowed_goal_sources[combo_name])
            for combo_name in allowed_goal_sources
        }

        self.action_root = self.root_dir / "action"
        self.rgb_root = self.root_dir / "rgb"

        self.episodes: List[Dict[str, Any]] = self._load_episode_jsons(self.action_root)
        self.samples: List[Dict[str, Any]] = self._build_sample_index(self.episodes)

    def __len__(self) -> int:
        return len(self.samples)

    # ---------------------------------------------------------------------
    # Index building
    # ---------------------------------------------------------------------
    def _load_episode_jsons(self, action_root: Path) -> List[Dict[str, Any]]:
        """
        Load all episode json files from:

            action/
                <goal_name>/
                    0000.json
                    0001.json
                    ...

        and return them as a flat episode list.
        """
        episodes: List[Dict[str, Any]] = []

        if not action_root.exists():
            raise FileNotFoundError(f"action root not found: {action_root}")

        goal_dirs = sorted([p for p in action_root.iterdir() if p.is_dir()])

        for goal_dir in goal_dirs:
            goal_name = goal_dir.name
            goal_name = goal_name.removeprefix("goal_")
            
            json_files = sorted(goal_dir.glob("*.json"))
            for json_path in json_files:
                with json_path.open("r", encoding="utf-8") as f:
                    ep = json.load(f)

                ep["language_instruction"] = f"go to {goal_name}"  # language_instruction 보정
                
                episodes.append(ep)

        return episodes

    def _build_sample_index(self, episodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build timestep-level samples from episode-level json.

        One dataset sample corresponds to one current timestep t inside one episode.
        Goal image / goal pose are selected later in __getitem__.
        """
        samples: List[Dict[str, Any]] = []

        for ep in episodes:
            traj = ep["trajectory"]
            if len(traj) < self.action_horizon + 2:
                continue

            # 보수적으로 너무 뒤쪽 frame은 제외하지 말고, padding 로직에 맡긴다.
            max_t = len(traj) - 1
            for t in range(max_t):
                samples.append(
                    {
                        "episode_id": ep["episode_id"],
                        "goal_name": ep["goal_name"],
                        "language_instruction": ep["language_instruction"],
                        "goal_pose": ep["goal_pose"], # world 좌표 !
                        "trajectory": traj,
                        "t": t,
                        "sample_id": f"{ep['goal_name']}_{ep['episode_id']}_{t}",
                    }
                )

        return samples

    # ---------------------------------------------------------------------
    # Image loading hooks
    # ---------------------------------------------------------------------
    def _frame_path(self, goal_name: str, episode_id: str, frame_index: int) -> Path:
        """
        Example:
            root/rgb/goal_marker1/0009/0000.png
        """
        return self.rgb_root / goal_name / str(episode_id) / f"rgb_{frame_index:04d}.png"

    def _load_rgb_frame(self, goal_name: str, episode_id: str, frame_index: int) -> np.ndarray:
        path = self._frame_path(goal_name, episode_id, frame_index)
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    # ---------------------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return math.atan2(math.sin(theta), math.cos(theta))

    def _world_to_relative_pose(
        self,
        robot_pose_world: Tuple[float, float, float],
        goal_pose_world: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert world-frame goal pose into robot-relative pose.

        Returns
        -------
        goal_pose_cos_sin : [4] = (x_rel, y_rel, cos(dyaw), sin(dyaw))
        obj_pose_norm     : [2] = (x_rel, y_rel)
        goal_distance     : float, metric distance in meters
        """
        xr, yr, yaw_r = robot_pose_world
        xg, yg, yaw_g = goal_pose_world

        dx = xg - xr
        dy = yg - yr

        x_rel = math.cos(yaw_r) * dx + math.sin(yaw_r) * dy
        y_rel = -math.sin(yaw_r) * dx + math.cos(yaw_r) * dy
        dyaw = self._wrap_angle(yaw_g - yaw_r)

        goal_distance = math.sqrt(x_rel ** 2 + y_rel ** 2)

        x_rel_norm = x_rel / self.metric_waypoint_spacing
        y_rel_norm = y_rel / self.metric_waypoint_spacing

        goal_pose_cos_sin = np.array(
            [x_rel_norm, y_rel_norm, math.cos(dyaw), math.sin(dyaw)],
            dtype=np.float32,
        )
        obj_pose_norm = np.array([x_rel_norm, y_rel_norm], dtype=np.float32)
        
        return goal_pose_cos_sin, obj_pose_norm, goal_distance

    def _global_traj_to_relative_actions(
        self,
        traj_pose_list: Sequence[Tuple[float, float, float]],
        start_idx: int,
        horizon: int,
        max_goal_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Convert stored global map poses into robot-relative waypoint action chunks.

        Output shape
        ------------
        [horizon, 4] = (x_rel, y_rel, cos(dyaw), sin(dyaw))

        max_goal_idx:
            If provided, generated actions are clamped so they never go beyond this index.
            Useful when the supervision goal is a future frame rather than the final destination.
        """
        x0, y0, yaw0 = traj_pose_list[start_idx]
        actions: List[List[float]] = []

        if max_goal_idx is None:
            max_goal_idx = len(traj_pose_list) - 1
        else:
            max_goal_idx = max(start_idx, min(max_goal_idx, len(traj_pose_list) - 1))

        for k in range(1, horizon + 1):
            j = start_idx + k * self.action_spacing
            j = min(j, max_goal_idx)

            xj, yj, yawj = traj_pose_list[j]

            dx = xj - x0
            dy = yj - y0

            x_rel = math.cos(yaw0) * dx + math.sin(yaw0) * dy
            y_rel = -math.sin(yaw0) * dx + math.cos(yaw0) * dy
            dyaw = self._wrap_angle(yawj - yaw0)

            actions.append(
                [
                    x_rel / self.metric_waypoint_spacing,
                    y_rel / self.metric_waypoint_spacing,
                    math.cos(dyaw),
                    math.sin(dyaw),
                ]
            )

        if len(actions) == 0:
            actions = [[0.0, 0.0, 1.0, 0.0] for _ in range(horizon)]
        while len(actions) < horizon:
            actions.append(actions[-1])

        return np.asarray(actions, dtype=np.float32)

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------
    def _resize_norm(self, img_chw: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        if img_chw.dtype != torch.float32:
            img_chw = img_chw.float()
        if img_chw.max() > 1.0:
            img_chw = img_chw / 255.0
        return torch.nn.functional.interpolate(
            img_chw.unsqueeze(0),
            size=out_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _determine_modality_id(
        self,
        satellite: bool,
        lan_prompt: bool,
        pose_goal: bool,
        image_goal: bool,
    ) -> torch.Tensor:
        if satellite and not lan_prompt and not pose_goal and not image_goal:
            return torch.as_tensor([0], dtype=torch.float32)
        elif satellite and not lan_prompt and pose_goal and not image_goal:
            return torch.as_tensor([1], dtype=torch.float32)
        elif satellite and not lan_prompt and not pose_goal and image_goal:
            return torch.as_tensor([2], dtype=torch.float32)
        elif satellite and not lan_prompt and pose_goal and image_goal:
            return torch.as_tensor([3], dtype=torch.float32)
        elif not satellite and not lan_prompt and pose_goal and not image_goal: # 4: pose 
            return torch.as_tensor([4], dtype=torch.float32)
        elif not satellite and not lan_prompt and pose_goal and image_goal:     # 5: pose + image
            return torch.as_tensor([5], dtype=torch.float32)
        elif not satellite and not lan_prompt and not pose_goal and image_goal: # 6: image
            return torch.as_tensor([6], dtype=torch.float32)
        elif not satellite and lan_prompt and not pose_goal and not image_goal: # 7: language
            return torch.as_tensor([7], dtype=torch.float32)
        elif not satellite and lan_prompt and pose_goal and not image_goal:     # 8: language + pose
            return torch.as_tensor([8], dtype=torch.float32)
        raise ValueError(
            f"Unsupported modality combination: satellite={satellite}, "
            f"lan_prompt={lan_prompt}, pose_goal={pose_goal}, image_goal={image_goal}"
        )

    def _sample_from_probs(self, probs: Dict[str, float]) -> str:
        r = random.random()
        cum = 0.0
        last_key = None

        for key, p in probs.items():
            cum += p
            last_key = key
            if r < cum:
                return key

        if last_key is None:
            raise ValueError(f"Failed to sample from empty probabilities: {probs}")

        return last_key

    def _sample_training_case(self) -> Dict[str, Any]:
        """
        Possible sampling outcomes:
            - destination pose                         -> modality 4 
            - future pose                              -> modality 4 
            
            - future image + future pose               -> modality 5
            - destination image + destination pose     -> modality 5
            
            - future image                             -> modality 6
            - destination image                        -> modality 6
            
            - language                                 -> modality 7
            
            - language + destination pose              -> modality 8
 
        """
        satellite = False  # NO satellite in my dataset

        combo = self._sample_from_probs(self.modality_combo_probs)
        allowed_sources = self.allowed_goal_sources[combo]

        restricted_goal_source_probs = {
            source_name: prob
            for source_name, prob in self.goal_source_probs.items()
            if source_name in allowed_sources
        }
        prob_sum = sum(restricted_goal_source_probs.values())
        if prob_sum <= 0.0:
            raise ValueError(
                f"No valid goal source probabilities for combo={combo}: "
                f"allowed_sources={allowed_sources}, goal_source_probs={self.goal_source_probs}"
            )

        restricted_goal_source_probs = {
            source_name: prob / prob_sum
            for source_name, prob in restricted_goal_source_probs.items()
        }
        goal_source = self._sample_from_probs(restricted_goal_source_probs)

        lan_prompt = combo in {"language_only", "language_pose"}
        pose_goal = combo in {"pose_only", "image_pose", "language_pose"}
        image_goal = combo in {"image_only", "image_pose"}

        modality_id = self._determine_modality_id(
            satellite=satellite,
            lan_prompt=lan_prompt,
            pose_goal=pose_goal,
            image_goal=image_goal,
        )

        if combo == "pose_only":
            case_name = "pose_only_future" if goal_source == "future" else "pose_only_destination"

        elif combo == "image_only":
            case_name = "image_only_future" if goal_source == "future" else "image_only_destination"

        elif combo == "language_only":
            case_name = "language_only_destination"

        elif combo == "image_pose":
            case_name = (
                "future_image_future_pose"
                if goal_source == "future"
                else "destination_image_destination_pose"
            )

        elif combo == "language_pose":
            case_name = "language_destination_pose"

        else:
            raise ValueError(f"Unknown modality combo: {combo}")

        return {
            "case_name": case_name,
            "satellite": satellite,
            "lan_prompt": lan_prompt,
            "pose_goal": pose_goal,
            "image_goal": image_goal,
            "goal_source": goal_source,
            "modality_id": modality_id,
        }
    def _normalize_language(self, goal_name: str, language_instruction: str) -> str:
        text = (language_instruction or "").strip().lower()
        if not text:
            text = goal_name.strip().lower()
        # uploaded examples use values like "goal_marker1" directly,
        # so convert to an instruction-like form.
        return f"go to {text}"

    def _build_prompt_and_labels(self, lang: str, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        current_action = actions[0]
        future_actions = actions[1:]

        current_action_string = self.action_tokenizer(current_action)
        future_actions_string = "".join(self.action_tokenizer(future_actions))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        if lang == "No language instruction":
            human_text = "No language instruction"
        else:
            human_text = f"What action should the robot take to {lang}?"

        conversation = [
            {"from": "human", "value": human_text},
            {"from": "gpt", "value": action_chunk_string},
        ]

        prompt_builder = self.prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(
            prompt_builder.get_prompt(),
            add_special_tokens=True,
        ).input_ids
        labels = list(input_ids)

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
            
        return input_ids, labels

    # ---------------------------------------------------------------------
    # Main sample creation
    # ---------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # TODO: context size 가 없네 ???        
        
        sample = self.samples[idx]

        episode_id = sample["episode_id"]
        goal_name = sample["goal_name"]
        language_instruction = sample["language_instruction"]
        traj = sample["trajectory"]
        t = sample["t"]
        final_goal_pose = sample["goal_pose"]
        sample_id = sample["sample_id"]

        # 1) current state/image
        cur_frame_idx = traj[t]["index"]
        current_img_np = self._load_rgb_frame(goal_name, episode_id, cur_frame_idx)
        current_world = (
            traj[t]["map_pose"]["x"],
            traj[t]["map_pose"]["y"],
            traj[t]["map_pose"]["yaw"],
        )

        # 2) choose goal image / goal pose source
        sample_case = self._sample_training_case()
        goal_source = sample_case["goal_source"]
        lan_prompt = sample_case["lan_prompt"]
        pose_goal = sample_case["pose_goal"]
        image_goal = sample_case["image_goal"]
        modality_id = sample_case["modality_id"]

        if goal_source == "future":
            # trajectory future frame as local waypoint-style goal
            max_future = min(len(traj) - 1, t + self.max_future_gap)
            min_future = min(len(traj) - 1, t + self.min_future_gap)
            if max_future <= min_future:
                g = min(len(traj) - 1, t + 1)
            else:
                g = random.randint(min_future, max_future)
            
            goal_frame_idx = traj[g]["index"]
            goal_img_np = self._load_rgb_frame(goal_name, episode_id, goal_frame_idx)
            goal_world = (
                traj[g]["map_pose"]["x"],
                traj[g]["map_pose"]["y"],
                traj[g]["map_pose"]["yaw"],
            )
            goal_image_type = "future_frame"

        elif goal_source == "destination":
            # use the last frame of the episode as destination image
            g = len(traj) - 5 # trajectory는 기록 됐는데, frame은 없을수도 있어서 -5 로 보수적으로 접근
            goal_frame_idx = traj[g]["index"]
            goal_img_np = self._load_rgb_frame(goal_name, episode_id, goal_frame_idx)
            goal_world = (
                final_goal_pose["x"],
                final_goal_pose["y"],
                final_goal_pose["yaw"],
            )
            goal_image_type = "destination"

        else:
            raise ValueError(f"Unsupported goal_source: {goal_source}")

        # 3) goal pose from world -> relative
        goal_pose_cos_sin, obj_pose_norm, goal_distance = self._world_to_relative_pose(
            current_world,
            goal_world,
        )

        # 4) build action chunk from global waypoints
        traj_pose_list = [
            (step["map_pose"]["x"], step["map_pose"]["y"], step["map_pose"]["yaw"])
            for step in traj
        ]

        max_action_goal_idx = g if goal_source == "future" else None

        actions_np = self._global_traj_to_relative_actions(
            traj_pose_list,
            start_idx=t,
            horizon=self.action_horizon,
            max_goal_idx=max_action_goal_idx,
        )
        actions = torch.as_tensor(actions_np, dtype=torch.float32)

        # 5) image transforms
        pil_img = Image.fromarray(current_img_np.astype(np.uint8)).resize(self.image_size_clip)
        pil_goal = Image.fromarray(goal_img_np.astype(np.uint8)).resize(self.image_size_clip)

        pixel_values = self.image_transform(pil_img)
        pixel_values_goal = self.image_transform(pil_goal)

        cur_image = self._resize_norm(
            torch.from_numpy(current_img_np).permute(2, 0, 1),
            self.image_size,
        )
        goal_image_8 = self._resize_norm(
            torch.from_numpy(goal_img_np).permute(2, 0, 1),
            self.image_size,
        )

        # 6) mask out unused modalities
        if not lan_prompt:
            language_instruction = "No language instruction"
            
        if not pose_goal:
            goal_pose_cos_sin = np.zeros(4, dtype=np.float32)
            obj_pose_norm = np.zeros(2, dtype=np.float32)
            
        if not image_goal:
            pil_goal = pil_img
            pixel_values_goal = torch.zeros_like(pixel_values_goal)
            goal_image_8 = torch.zeros_like(goal_image_8)

        # 7) optional horizontal flip
        if self.use_flip_aug and random.random() > 0.5:
            cur_image = torch.flip(cur_image, [2])
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            pixel_values = self.image_transform(pil_img)

            if image_goal:
                goal_image_8 = torch.flip(goal_image_8, [2])
                pil_goal = pil_goal.transpose(Image.FLIP_LEFT_RIGHT)
                pixel_values_goal = self.image_transform(pil_goal)

            if pose_goal:
                goal_pose_cos_sin[1] = -goal_pose_cos_sin[1]
                goal_pose_cos_sin[3] = -goal_pose_cos_sin[3]
                obj_pose_norm[1] = -obj_pose_norm[1]

            # action labels are waypoint-style [x, y, cos(yaw), sin(yaw)] chunks
            if actions.shape[-1] >= 4:
                actions[:, 1] = -actions[:, 1]
                actions[:, 3] = -actions[:, 3]

        # 8) build prompt + labels
        input_ids, labels = self._build_prompt_and_labels(language_instruction, actions)

        return dict(
            pixel_values=pixel_values,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            dataset_name="goto/sim",
            modality_id=modality_id,
            actions=actions,
            action_select_mask=torch.tensor(1.0),
            goal_pose=torch.as_tensor(goal_pose_cos_sin, dtype=torch.float32),
            obj_pose_norm=torch.as_tensor(obj_pose_norm, dtype=torch.float32),
            img_PIL=pil_img,
            gimg_PIL=pil_goal,
            cur_image=cur_image,
            goal_image_8=goal_image_8,
            temp_dist=goal_distance,
            lan_prompt=language_instruction,
            sample_id=sample_id,
            goal_image_type=goal_image_type,
        )