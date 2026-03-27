import argparse
import math
import random
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image

from prismatic.vla.datasets.goto_sim_dataset import GotoSim_Dataset


# plt.rcParams.update({
#     "font.size": 30,
#     "axes.titlesize": 30,
#     "axes.labelsize": 28,
#     "figure.titlesize": 32,
#     "xtick.labelsize": 26,
#     "ytick.labelsize": 26,
# })


# ---------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------
class _IdentityImageTransform:
    def __call__(self, image):
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = torch.from_numpy(arr).permute(2, 0, 1) / 255.0
        else:
            arr = torch.from_numpy(arr)[None, ...] / 255.0
        return arr


class _DummyActionTokenizer:
    def __call__(self, action):
        if isinstance(action, torch.Tensor):
            if action.ndim == 1:
                return "<action>"
            return ["<action>" for _ in range(action.shape[0])]
        return "<action>"


class _DummyPromptBuilder:
    def __init__(self, *_args, **_kwargs):
        self.parts = []

    def add_turn(self, role, value):
        self.parts.append((role, value))

    def get_prompt(self):
        return "\n".join(f"{r}: {v}" for r, v in self.parts)


class _DummyBaseTokenizer:
    class _Result:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    def __call__(self, text, add_special_tokens=True):
        ids = [1] + [2] * max(1, len(str(text).split()))
        if add_special_tokens:
            ids.append(3)
        return self._Result(ids)


# ---------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------
class VisualGotoSimDataset(GotoSim_Dataset):
    def _get_frame_id(self, traj: Sequence[Dict[str, Any]], t: int) -> int:
        step = traj[t]
        if isinstance(step, dict) and "index" in step:
            return int(step["index"])
        return int(t)

    def _frame_path_from_traj(self, goal_name: str, episode_id: str, traj: Sequence[Dict[str, Any]], t: int):
        frame_id = self._get_frame_id(traj, t)
        return self.rgb_root / goal_name / str(episode_id) / f"rgb_{frame_id:04d}.png"

    def _load_rgb_frame_by_t(self, goal_name: str, episode_id: str, traj: Sequence[Dict[str, Any]], t: int) -> np.ndarray:
        path = self._frame_path_from_traj(goal_name, episode_id, traj, t)
        if not path.exists():
            raise FileNotFoundError(
                f"RGB frame not found: {path} "
                f"(goal_name={goal_name}, episode_id={episode_id}, t={t}, "
                f"frame_id={self._get_frame_id(traj, t)})"
            )
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    def get_visual_sample(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        episode_id = sample["episode_id"]
        goal_name = sample["goal_name"]
        traj = sample["trajectory"]
        t = sample["t"]
        final_goal_pose = sample["goal_pose"]

        current_frame_id = self._get_frame_id(traj, t)
        current_img = self._load_rgb_frame_by_t(goal_name, episode_id, traj, t)
        current_world = (
            traj[t]["map_pose"]["x"],
            traj[t]["map_pose"]["y"],
            traj[t]["map_pose"]["yaw"],
        )

        case = self._sample_training_case()
        goal_source = case["goal_source"]
        lan_prompt = case["lan_prompt"]
        pose_goal = case["pose_goal"]
        image_goal = case["image_goal"]

        if goal_source == "future":
            max_future = min(len(traj) - 1, t + self.max_future_gap)
            min_future = min(len(traj) - 1, t + self.min_future_gap)
            if max_future <= min_future:
                g = min(len(traj) - 1, t + 1)
            else:
                g = random.randint(min_future, max_future)

            goal_frame_id = self._get_frame_id(traj, g)
            goal_img = self._load_rgb_frame_by_t(goal_name, episode_id, traj, g)
            goal_world = (
                traj[g]["map_pose"]["x"],
                traj[g]["map_pose"]["y"],
                traj[g]["map_pose"]["yaw"],
            )
            goal_image_type = "future_frame"
            max_action_goal_idx = g

        elif goal_source == "destination":
            g = len(traj) - 1
            goal_frame_id = self._get_frame_id(traj, g)
            goal_img = self._load_rgb_frame_by_t(goal_name, episode_id, traj, g)
            goal_world = (
                final_goal_pose["x"],
                final_goal_pose["y"],
                final_goal_pose["yaw"],
            )
            goal_image_type = "destination"
            max_action_goal_idx = None
        else:
            raise ValueError(f"Unsupported goal_source: {goal_source}")

        goal_pose_norm, obj_pose_norm, goal_distance = self._world_to_relative_pose(
            current_world,
            goal_world,
        )

        traj_pose_list = [
            (
                step["map_pose"]["x"],
                step["map_pose"]["y"],
                step["map_pose"]["yaw"],
            )
            for step in traj
        ]

        try:
            actions_norm = self._global_traj_to_relative_actions(
                traj_pose_list,
                start_idx=t,
                horizon=self.action_horizon,
                max_goal_idx=max_action_goal_idx,
            )
        except TypeError:
            actions_norm = self._global_traj_to_relative_actions(
                traj_pose_list,
                start_idx=t,
                horizon=self.action_horizon,
            )

        actions_m = actions_norm.copy()
        actions_m[:, :2] *= self.metric_waypoint_spacing

        goal_pose_m = goal_pose_norm.copy()
        goal_pose_m[:2] *= self.metric_waypoint_spacing
        obj_pose_m = obj_pose_norm.copy() * self.metric_waypoint_spacing

        language_instruction = sample["language_instruction"] if lan_prompt else "No language instruction"

        if not pose_goal:
            goal_pose_norm = np.zeros(4, dtype=np.float32)
            obj_pose_norm = np.zeros(2, dtype=np.float32)
            goal_pose_m = np.zeros(4, dtype=np.float32)
            obj_pose_m = np.zeros(2, dtype=np.float32)

        if not image_goal:
            goal_img_masked = np.zeros_like(goal_img)
        else:
            goal_img_masked = goal_img

        return {
            "sample_id": sample["sample_id"],
            "goal_name": goal_name,
            "episode_id": episode_id,
            "t": t,
            "g": g,
            "traj_len": len(traj),
            "current_frame_id": current_frame_id,
            "goal_frame_id": goal_frame_id,
            "current_img": current_img,
            "goal_img": goal_img_masked,
            "goal_img_raw": goal_img,
            "goal_source": goal_source,
            "goal_image_type": goal_image_type,
            "language_instruction": language_instruction,
            "case": case,
            "goal_distance_m": goal_distance,
            "actions_norm": actions_norm,
            "actions_m": actions_m,
            "goal_pose_norm": goal_pose_norm,
            "goal_pose_m": goal_pose_m,
            "obj_pose_norm": obj_pose_norm,
            "obj_pose_m": obj_pose_m,
            "current_world": current_world,
            "goal_world": goal_world,
        }


# ---------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------
def draw_robot(ax, x=0.0, y=0.0, yaw=0.0, scale=0.16, label=None):
    dx = scale * math.cos(yaw)
    dy = scale * math.sin(yaw)
    ax.arrow(
        x, y, dx, dy,
        length_includes_head=True,
        head_width=max(scale * 0.35, 0.05),
        head_length=max(scale * 0.45, 0.07),
    )
    if label is not None:
        ax.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=22)


def add_text_block(ax, title, lines, title_size=25, text_size=13, x=0.0):
    ax.clear()
    ax.axis("off")
    ax.text(x, 1.0, title, va="top", ha="left", fontsize=title_size, weight="bold", transform=ax.transAxes)
    ax.text(
        x, 0.76, "\n".join(lines),
        va="top", ha="left",
        family="monospace", fontsize=text_size,
        linespacing=1.32, transform=ax.transAxes
    )


def draw_status_chip(ax, x, y, enabled, label, width=0.12, height=0.10):
    patch = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        fill=enabled,
        hatch=None if enabled else "///",
        linewidth=1.0,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + width + 0.025, y + height / 2,
        label,
        va="center", ha="left",
        fontsize=23, weight="bold",
        transform=ax.transAxes,
    )


def add_conditioning_table(ax, sample):
    ax.clear()
    ax.axis("off")
    case = sample["case"]
    modality_id = int(case["modality_id"].item()) if hasattr(case["modality_id"], "item") else int(case["modality_id"])

    ax.text(
        0.0, 1.0,
        f"Conditioning summary   |   modality_id = {modality_id}   |   case = {case['case_name']}",
        va="top", ha="left",
        family="monospace", fontsize=25,
        transform=ax.transAxes
    )

    rows = [
        ("Current image", True, [
            f"frame_id : {sample['current_frame_id']}",
            f"t index  : {sample['t']}",
        ]),
        ("Goal image", case["image_goal"], [
            f"frame_id : {sample['goal_frame_id']}",
            f"g index  : {sample['g']}",
            f"source   : {sample['goal_source']} / {sample['goal_image_type']}",
        ]),
        ("Goal pose", case["pose_goal"], [
            (f"x, y     : {sample['goal_pose_m'][0]:.3f}, {sample['goal_pose_m'][1]:.3f}" if case["pose_goal"] else "x, y     : masked"),
            (f"yaw      : {math.atan2(sample['goal_pose_norm'][3], sample['goal_pose_norm'][2]):.3f}" if case["pose_goal"] else "yaw      : masked"),
            (f"distance : {sample['goal_distance_m']:.3f} m" if case["pose_goal"] else ""),
        ]),
        ("Language", case["lan_prompt"], [
            f"text     : {sample['language_instruction']}" if case["lan_prompt"] else "text     : masked",
        ]),
    ]

    base_y = 0.80
    line_step = 0.062
    gap = 0.07
    x_info = 0.30

    current_y = base_y
    for label, enabled, info_lines in rows:
        clean = [ln for ln in info_lines if ln]
        n_lines = max(1, len(clean))
        chip_y = current_y - 0.05
        draw_status_chip(ax, 0.02, chip_y, enabled, label, width=0.10, height=0.09)

        ax.text(
            x_info, current_y,
            "\n".join(clean),
            va="top", ha="left",
            family="monospace", fontsize=23,
            linespacing=1.28, transform=ax.transAxes
        )
        current_y -= (n_lines * line_step + gap)


def add_action_table(ax, sample):
    ax.clear()
    ax.axis("off")
    action_xy = sample["actions_m"][:, :2]
    yaw_from_actions = np.arctan2(sample["actions_norm"][:, 3], sample["actions_norm"][:, 2])
    endpoint_dist = float(np.linalg.norm(action_xy[-1])) if len(action_xy) > 0 else 0.0

    lines = ["Action steps  (local frame)", ""]
    lines.append(f"{'k':>2}   {'x [m]':>10}   {'y [m]':>10}   {'yaw [rad]':>10}")
    lines.append("-" * 42)
    for i, (xy, yaw) in enumerate(zip(action_xy, yaw_from_actions), start=1):
        lines.append(f"{i:>2}   {xy[0]:>10.3f}   {xy[1]:>10.3f}   {yaw:>10.3f}")
    lines.extend([
        "",
        f"endpoint distance : {endpoint_dist:.3f} m",
        f"goal distance     : {sample['goal_distance_m']:.3f} m",
    ])

    ax.text(
        0.0, 1.0, "\n".join(lines),
        va="top", ha="left",
        family="monospace", fontsize=23,
        linespacing=1.20, transform=ax.transAxes
    )


def draw_trajectory(ax, sample):
    ax.clear()
    ax.grid(True, alpha=0.28)

    goal_xy = sample["goal_pose_m"][:2]
    action_xy = sample["actions_m"][:, :2]

    path_xy = np.vstack([[0.0, 0.0], action_xy])
    plot_x = path_xy[:, 1]
    plot_y = path_xy[:, 0]

    ax.plot(plot_x, plot_y, linewidth=2.6)
    ax.scatter(plot_x[1:], plot_y[1:], s=36)
    draw_robot(ax, 0.0, 0.0, math.pi / 2, scale=0.14, label="start")

    all_x = plot_x.tolist()
    all_y = plot_y.tolist()

    if sample["case"]["pose_goal"]:
        goal_plot_x = goal_xy[1]
        goal_plot_y = goal_xy[0]
        ax.scatter([goal_plot_x], [goal_plot_y], marker="*", s=180)
        ax.plot([0.0, goal_plot_x], [0.0, goal_plot_y], linestyle="--", linewidth=1.4)
        ax.annotate("goal", (goal_plot_x, goal_plot_y), xytext=(6, 6), textcoords="offset points", fontsize=22)
        all_x.append(goal_plot_x)
        all_y.append(goal_plot_y)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    span = max(x_max - x_min, y_max - y_min, 1.0)
    pad = 0.22 * span + 0.18
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    ax.set_xlim(cx - span / 2 - pad, cx + span / 2 + pad)
    ax.set_ylim(cy - span / 2 - pad, cy + span / 2 + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("lateral [m]", fontsize=23)
    ax.set_ylabel("forward [m]", fontsize=23)
    ax.set_title("Action trajectory (robot forward = up)", fontsize=25)


# ---------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------
class InteractiveVisualizer:
    def __init__(self, dataset: VisualGotoSimDataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
        self.ptr = 0

        self.fig = plt.figure(figsize=(18, 12))
        self.gs = self.fig.add_gridspec(
            4, 2,
            width_ratios=[1.0, 1.0],
            height_ratios=[1.0, 0.45, 1.15, 1.20],
            hspace=0.34,
            wspace=0.22,
        )

        self.ax_cur = self.fig.add_subplot(self.gs[0, 0])
        self.ax_goal = self.fig.add_subplot(self.gs[0, 1])

        self.ax_cur_info = self.fig.add_subplot(self.gs[1, 0])
        self.ax_goal_info = self.fig.add_subplot(self.gs[1, 1])

        self.ax_cond = self.fig.add_subplot(self.gs[2, :])

        self.bottom_gs = self.gs[3, :].subgridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.20)
        self.ax_traj = self.fig.add_subplot(self.bottom_gs[0, 0])
        self.ax_actions = self.fig.add_subplot(self.bottom_gs[0, 1])

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.render_current()

    def render_current(self):
        sample = self.dataset.get_visual_sample(self.indices[self.ptr])

        self.ax_cur.clear()
        self.ax_goal.clear()

        self.ax_cur.imshow(sample["current_img"])
        self.ax_cur.set_title("Current image", fontsize=26)
        self.ax_cur.axis("off")

        self.ax_goal.imshow(sample["goal_img"])
        self.ax_goal.set_title("Goal image", fontsize=26)
        self.ax_goal.axis("off")

        current_world = sample["current_world"]
        goal_world = sample["goal_world"]

        add_text_block(
            self.ax_cur_info,
            "Current image details",
            [
                f"frame_id : {sample['current_frame_id']}",
                f"t index  : {sample['t']}",
                f"world x,y: {current_world[0]:.3f}, {current_world[1]:.3f}",
                f"world yaw: {current_world[2]:.3f}",
            ],
            title_size=25,
            text_size=23,
        )

        add_text_block(
            self.ax_goal_info,
            "Goal image details",
            [
                f"frame_id : {sample['goal_frame_id']}",
                f"g index  : {sample['g']}",
                f"source   : {sample['goal_source']}",
                f"type     : {sample['goal_image_type']}",
                f"world x,y: {goal_world[0]:.3f}, {goal_world[1]:.3f}",
                f"world yaw: {goal_world[2]:.3f}",
            ],
            title_size=25,
            text_size=23,
        )

        add_conditioning_table(self.ax_cond, sample)
        draw_trajectory(self.ax_traj, sample)
        add_action_table(self.ax_actions, sample)

        self.fig.suptitle(
            f"GotoSim interactive visualizer   |   sample {self.ptr + 1}/{len(self.indices)}   "
            f"|   idx={self.indices[self.ptr]}   |   [Right/Space/N] next   [Left/P] prev   [R] reshuffle   [Q] quit",
            fontsize=26,
        )
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ("right", " ", "n", "enter"):
            self.ptr = (self.ptr + 1) % len(self.indices)
            self.render_current()
        elif event.key in ("left", "p", "backspace"):
            self.ptr = (self.ptr - 1) % len(self.indices)
            self.render_current()
        elif event.key == "r":
            random.shuffle(self.indices)
            self.ptr = 0
            self.render_current()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def build_dataset(args):
    return VisualGotoSimDataset(
        root_dir=args.root_dir,
        image_transform=_IdentityImageTransform(),
        action_tokenizer=_DummyActionTokenizer(),
        prompt_builder_fn=_DummyPromptBuilder,
        base_tokenizer=_DummyBaseTokenizer(),
        image_size=(224, 224),
        image_size_clip=(224, 224),
        metric_waypoint_spacing=args.metric_waypoint_spacing,
        action_horizon=args.action_horizon,
        action_spacing=args.action_spacing,
        min_future_gap=args.min_future_gap,
        max_future_gap=args.max_future_gap,
        predict_stop_token=False,
        use_flip_aug=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--metric-waypoint-spacing", type=float, default=0.25)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--action-spacing", type=int, default=5)
    parser.add_argument("--min-future-gap", type=int, default=12)
    parser.add_argument("--max-future-gap", type=int, default=60)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = build_dataset(args)
    if len(dataset.samples) == 0:
        raise RuntimeError("No valid samples found. Check dataset root or sampling conditions.")

    num = min(args.num_samples, len(dataset.samples))
    indices = random.sample(range(len(dataset.samples)), k=num)

    viewer = InteractiveVisualizer(dataset, indices)
    plt.show()


if __name__ == "__main__":
    main()
    
"""
python3 data_sample_visualizer.py \
  --root-dir /nas/sujinkim/data/goto/sim/20260323 \
  --num-samples 100
"""