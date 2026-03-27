
import os
import json
import math
import random
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# helpers
# -----------------------------

def wrap_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def stats(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def round_stats_dict(d, ndigits=2):
    if d is None:
        return None
    out = {}
    for k, v in d.items():
        out[k] = round(float(v), ndigits)
    return out


def normalize_probs(d):
    s = sum(d.values())
    if s <= 0:
        raise ValueError("Probability sum must be positive.")
    return {k: v / s for k, v in d.items()}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# dataset loading
# -----------------------------

def list_episode_files(root_dir):
    action_root = os.path.join(root_dir, "action")
    if not os.path.isdir(action_root):
        raise ValueError(f"Could not find action directory: {action_root}")

    files = []
    for cur_dir, _, filenames in os.walk(action_root):
        for name in filenames:
            if name.endswith(".json"):
                files.append(os.path.join(cur_dir, name))
    files.sort()
    return files


def build_sample_index(episodes):
    idx = []
    for ep_i, ep in enumerate(episodes):
        traj = ep["trajectory"]
        for t in range(len(traj) - 1):
            idx.append((ep_i, t))
    return idx


# -----------------------------
# sampling policy
# -----------------------------

def sample_from_probs(probs, rng):
    r = rng.random()
    cum = 0.0
    last_key = None
    for k, p in probs.items():
        cum += p
        last_key = k
        if r < cum:
            return k
    return last_key


def determine_modality_id(lan_prompt, pose_goal, image_goal):
    if pose_goal and not image_goal and not lan_prompt:
        return 4
    if pose_goal and image_goal and not lan_prompt:
        return 5
    if image_goal and not pose_goal and not lan_prompt:
        return 6
    if lan_prompt and not pose_goal and not image_goal:
        return 7
    if lan_prompt and pose_goal and not image_goal:
        return 8
    raise ValueError(
        f"Unsupported combo: lan={lan_prompt}, pose={pose_goal}, image={image_goal}"
    )


def sample_training_case(goal_source_probs, modality_probs, allowed_goal_sources, rng):
    combo = sample_from_probs(modality_probs, rng)
    allowed = allowed_goal_sources[combo]
    restricted = {k: v for k, v in goal_source_probs.items() if k in allowed}
    restricted = normalize_probs(restricted)
    goal_source = sample_from_probs(restricted, rng)

    lan_prompt = combo in {"language_only", "language_pose"}
    pose_goal = combo in {"pose_only", "image_pose", "language_pose"}
    image_goal = combo in {"image_only", "image_pose"}
    modality_id = determine_modality_id(lan_prompt, pose_goal, image_goal)

    return {
        "combo": combo,
        "goal_source": goal_source,
        "lan_prompt": lan_prompt,
        "pose_goal": pose_goal,
        "image_goal": image_goal,
        "modality_id": modality_id,
    }


def outcome_name(combo, goal_source):
    if combo == "pose_only":
        return f"{goal_source} pose"
    if combo == "image_pose":
        return f"{goal_source} image + {goal_source} pose"
    if combo == "image_only":
        return f"{goal_source} image"
    if combo == "language_only":
        return "language"
    if combo == "language_pose":
        return "language + destination pose"
    return f"{goal_source}__{combo}"


def compute_sampling_outcome_comparison(
    goal_source_probs, modality_probs, allowed_goal_sources, num_draws, rng, ndigits=2
):
    expected = {}
    for combo, combo_prob in modality_probs.items():
        allowed = allowed_goal_sources[combo]
        restricted = {k: v for k, v in goal_source_probs.items() if k in allowed}
        restricted = normalize_probs(restricted)
        for goal_source, p in restricted.items():
            name = outcome_name(combo, goal_source)
            expected[name] = float(combo_prob * p)

    sampled_counter = Counter()
    for _ in range(num_draws):
        case = sample_training_case(
            goal_source_probs=goal_source_probs,
            modality_probs=modality_probs,
            allowed_goal_sources=allowed_goal_sources,
            rng=rng,
        )
        sampled_counter[outcome_name(case["combo"], case["goal_source"])] += 1

    all_names = []
    for name in expected.keys():
        if name not in all_names:
            all_names.append(name)
    for name in sampled_counter.keys():
        if name not in all_names:
            all_names.append(name)

    comparison = {}
    for name in all_names:
        comparison[name] = {
            "expected": round(expected.get(name, 0.0), ndigits),
            "sampled": round(sampled_counter.get(name, 0) / num_draws, ndigits),
        }
    return comparison


# -----------------------------
# geometry
# -----------------------------

def world_to_relative(robot, target):
    xr, yr, yawr = robot
    xt, yt, yawt = target

    dx = xt - xr
    dy = yt - yr

    x_rel = math.cos(yawr) * dx + math.sin(yawr) * dy
    y_rel = -math.sin(yawr) * dx + math.cos(yawr) * dy

    dyaw = wrap_angle(yawt - yawr)
    return x_rel, y_rel, dyaw


def compute_actions(traj, start_idx, horizon, spacing, metric_spacing):
    x0, y0, yaw0 = traj[start_idx]
    actions = []

    for k in range(1, horizon + 1):
        j = min(start_idx + k * spacing, len(traj) - 1)

        xj, yj, yawj = traj[j]

        dx = xj - x0
        dy = yj - y0

        x_rel = math.cos(yaw0) * dx + math.sin(yaw0) * dy
        y_rel = -math.sin(yaw0) * dx + math.cos(yaw0) * dy

        dyaw = wrap_angle(yawj - yaw0)

        actions.append([
            x_rel / metric_spacing,
            y_rel / metric_spacing,
            math.cos(dyaw),
            math.sin(dyaw),
        ])

    return np.asarray(actions, dtype=np.float32)


# -----------------------------
# visualization
# -----------------------------

def add_stat_text(ax, arr, unit=""):
    s = stats(arr)
    txt = (
        f"min={s['min']:.2f}{unit}\n"
        f"p25={s['p25']:.2f}{unit}\n"
        f"p50={s['p50']:.2f}{unit}\n"
        f"mean={s['mean']:.2f}{unit}\n"
        f"p75={s['p75']:.2f}{unit}\n"
        f"p95={s['p95']:.2f}{unit}\n"
        f"max={s['max']:.2f}{unit}"
    )
    ax.text(
        0.98, 0.95, txt,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def draw_boxplot(ax, values, title, ylabel):
    ax.boxplot(values, vert=True, showfliers=False)
    ax.set_title(title)
    ax.set_xticks([1])
    ax.set_xticklabels([ylabel])
    ax.grid(True, axis="y", alpha=0.3)
    add_stat_text(ax, values)


def plot_sampling(path, comp):
    names = list(comp.keys())
    expected = [comp[k]["expected"] for k in names]
    sampled = [comp[k]["sampled"] for k in names]

    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x - w / 2, expected, w, label="expected")
    ax.bar(x + w / 2, sampled, w, label="sampled")

    ax.set_title("Sampling Outcome Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("probability")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for i, (e, s) in enumerate(zip(expected, sampled)):
        ax.text(i - w / 2, e, f"{e:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, s, f"{s:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_one_page_boxplots(path, args, waypoint_dist_m, step1_dist, action_x_norm, future_goal_dist):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    draw_boxplot(
        axes[0, 0],
        waypoint_dist_m,
        f"Action Horizon = {args.action_horizon}",
        "Predicted Dist (m)",
    )
    draw_boxplot(
        axes[0, 1],
        step1_dist,
        f"Action Spacing = {args.action_spacing}",
        "Step-1 Dist (m)",
    )
    draw_boxplot(
        axes[1, 0],
        action_x_norm,
        f"Metric Spacing = {args.metric_waypoint_spacing}",
        "x_rel Norm",
    )
    draw_boxplot(
        axes[1, 1],
        future_goal_dist,
        f"Future Gap = [{args.min_future_gap}, {args.max_future_gap}]",
        "Future Goal Dist (m)",
    )

    fig.suptitle("Dataset Brief Summary - One Page Boxplots", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


# -----------------------------
# main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--out_dir", type=str, default="dataset_brief_report")

    parser.add_argument("--metric_waypoint_spacing", type=float, default=0.25)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--action_spacing", type=int, default=5)
    parser.add_argument("--min_future_gap", type=int, default=12)
    parser.add_argument("--max_future_gap", type=int, default=24)

    parser.add_argument(
        "--goal_source_probs",
        type=str,
        default='{"future": 0.4, "destination": 0.6}',
    )
    parser.add_argument(
        "--modality_probs",
        type=str,
        default='{"pose_only": 0.10, "image_only": 0.20, "language_only": 0.05, "image_pose": 0.45, "language_pose": 0.20}',
    )
    parser.add_argument(
        "--allowed_goal_sources",
        type=str,
        default='{"pose_only": ["future", "destination"], "image_only": ["future", "destination"], "language_only": ["destination"], "image_pose": ["future", "destination"], "language_pose": ["destination"]}',
    )

    parser.add_argument("--max_samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / "dataset_brief_summary.json"
    viz_file = out_dir / "one_page_boxplots.png"
    sampling_file = out_dir / "sampling_outcome_comparison.png"
    readme_file = out_dir / "README.txt"

    dataset_rng = random.Random(args.seed)
    sampling_rng = random.Random(args.seed)

    goal_source_probs = normalize_probs(json.loads(args.goal_source_probs))
    modality_probs = normalize_probs(json.loads(args.modality_probs))
    allowed_goal_sources = json.loads(args.allowed_goal_sources)

    files = list_episode_files(args.root_dir)
    episodes = [load_json(f) for f in files]
    sample_index = build_sample_index(episodes)

    n = min(args.max_samples, len(sample_index))
    if n == 0:
        raise ValueError("No samples found.")

    chosen = dataset_rng.sample(sample_index, n)

    waypoint_dist_m = []
    step1_dist = []
    action_x_norm = []
    future_goal_dist = []

    for ep_i, t in chosen:
        ep = episodes[ep_i]

        traj = [
            (s["map_pose"]["x"], s["map_pose"]["y"], s["map_pose"]["yaw"])
            for s in ep["trajectory"]
        ]

        robot = traj[t]

        lo = min(len(traj) - 1, t + args.min_future_gap)
        hi = min(len(traj) - 1, t + args.max_future_gap)

        if hi <= lo:
            g = min(len(traj) - 1, t + 1)
        else:
            g = dataset_rng.randint(lo, hi)

        goal = traj[g]

        gx, gy, _ = world_to_relative(robot, goal)
        future_goal_dist.append(math.sqrt(gx * gx + gy * gy))

        actions = compute_actions(
            traj=traj,
            start_idx=t,
            horizon=args.action_horizon,
            spacing=args.action_spacing,
            metric_spacing=args.metric_waypoint_spacing,
        )

        dist_norm = np.sqrt(actions[:, 0] ** 2 + actions[:, 1] ** 2)
        dist_m = dist_norm * args.metric_waypoint_spacing

        waypoint_dist_m.extend(dist_m.tolist())
        step1_dist.append(float(dist_m[0]))
        action_x_norm.extend(actions[:, 0].tolist())

    sampling_comp = compute_sampling_outcome_comparison(
        goal_source_probs=goal_source_probs,
        modality_probs=modality_probs,
        allowed_goal_sources=allowed_goal_sources,
        num_draws=n,
        rng=sampling_rng,
        ndigits=2,
    )

    summary = {
        "config": {
            "root_dir": args.root_dir,
            "out_dir": str(out_dir),
            "num_episode_files": len(files),
            "num_samples_indexed": len(sample_index),
            "num_samples_analyzed": n,
            "seed": args.seed,
        },
        "action_horizon": {
            "value": args.action_horizon,
            "predicted_distance_m": round_stats_dict(stats(waypoint_dist_m), 2),
        },
        "action_spacing": {
            "value": args.action_spacing,
            "step1_waypoint_distance_m": round_stats_dict(stats(step1_dist), 2),
            "note": "Distance to the first predicted waypoint (k=1).",
        },
        "metric_waypoint_spacing": {
            "value": args.metric_waypoint_spacing,
            "action_prediction_x_rel_norm": round_stats_dict(stats(action_x_norm), 2),
            "note": "Normalized x_rel values after division by metric_waypoint_spacing.",
        },
        "future_gap": {
            "min": args.min_future_gap,
            "max": args.max_future_gap,
            "future_goal_distance_m": round_stats_dict(stats(future_goal_dist), 2),
        },
        "sampling_outcome_comparison": sampling_comp,
    }

    with open(outfile, "w") as f:
        json.dump(summary, f, indent=2)

    plot_one_page_boxplots(
        str(viz_file),
        args,
        waypoint_dist_m,
        step1_dist,
        action_x_norm,
        future_goal_dist,
    )
    plot_sampling(str(sampling_file), sampling_comp)

    with open(readme_file, "w") as f:
        f.write("Generated files\n")
        f.write("===============\n\n")
        f.write("- dataset_brief_summary.json\n")
        f.write("- one_page_boxplots.png\n")
        f.write("- sampling_outcome_comparison.png\n\n")
        f.write("one_page_boxplots.png contains 4 boxplots in one page:\n")
        f.write("1. action horizon predicted distance (m)\n")
        f.write("2. action spacing step-1 distance (m)\n")
        f.write("3. metric spacing normalized x_rel values\n")
        f.write("4. future gap goal distance (m)\n")

    print(f"saved folder -> {out_dir}")
    print(f"saved json -> {outfile}")
    print(f"saved figure -> {viz_file}")
    print(f"saved figure -> {sampling_file}")


if __name__ == "__main__":
    main()
    
"""
python3 dataset_stats.py \
  --root_dir /nas/sujinkim/data/goto/sim/20260323 \
  --out_dir dataset_brief_report \
  --metric_waypoint_spacing 0.25 \
  --action_horizon 8 \
  --action_spacing 5 \
  --min_future_gap 12 \
  --max_future_gap 60 \
  --goal_source_probs '{"future": 0.6, "destination": 0.4}' \
  --modality_probs '{"pose_only": 0.10, "image_pose": 0.45, "image_only": 0.20, "language_only": 0.05,  "language_pose": 0.20}' \
  --allowed_goal_sources '{"pose_only": ["future", "destination"], "image_pose": ["future", "destination"], "image_only": ["future", "destination"], "language_only": ["destination"], "language_pose": ["destination"]}' \
  --seed 0
"""