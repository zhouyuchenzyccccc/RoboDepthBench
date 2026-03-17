"""

python scripts/visualize_saved_dataset.py \
  --data_path /inspire/qb-ilm/project/wuliqifa/public/zyc/dataset/pfp_state_recon/unplug_charger/train \
  --out_dir /inspire/qb-ilm/project/wuliqifa/public/zyc/dataset/pfp_state_recon/unplug_charger/train/quick_vis \
  --episode 0 \
  --start_step 0 \
  --num_steps 10 \
  --camera_idx 0 \
  --pcd_max_points 3000

"""
import argparse
from pathlib import Path

import numpy as np

from pfp.data.replay_buffer import RobotReplayBuffer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a subset of a saved PointFlowMatch zarr dataset."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to zarr dataset split")
    parser.add_argument("--out_dir", type=str, default=None, help="Output folder for visualizations")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--start_step", type=int, default=0, help="First step index")
    parser.add_argument("--num_steps", type=int, default=5, help="Number of steps to export")
    parser.add_argument(
        "--camera_idx",
        type=int,
        default=0,
        help="Camera index used for stacked keys like images/depth_sensor",
    )
    parser.add_argument(
        "--pcd_max_points",
        type=int,
        default=3000,
        help="Maximum number of point cloud points to draw",
    )
    return parser.parse_args()


def _get_step_keys(episode: dict[str, np.ndarray]) -> list[str]:
    return sorted(list(episode.keys()))


def _pick_rgb(episode: dict[str, np.ndarray], step_idx: int, camera_idx: int) -> np.ndarray | None:
    if "images" in episode:
        imgs = episode["images"][step_idx]
        if imgs.ndim == 4:
            return imgs[camera_idx]

    camera_names = ["right_shoulder", "left_shoulder", "overhead", "front", "wrist"]
    if 0 <= camera_idx < len(camera_names):
        key = f"rgb_{camera_names[camera_idx]}"
        if key in episode:
            return episode[key][step_idx]
    return None


def _pick_depth(episode: dict[str, np.ndarray], step_idx: int, camera_idx: int) -> np.ndarray | None:
    if "depth_sensor" in episode:
        depth = episode["depth_sensor"][step_idx]
        if depth.ndim == 3:
            return depth[camera_idx]

    camera_names = ["right_shoulder", "left_shoulder", "overhead", "front", "wrist"]
    if 0 <= camera_idx < len(camera_names):
        key = f"depth_sensor_{camera_names[camera_idx]}"
        if key in episode:
            return episode[key][step_idx]
    return None


def _pick_pcd(episode: dict[str, np.ndarray], step_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    pcd_xyz = episode["pcd_xyz"][step_idx] if "pcd_xyz" in episode else None
    pcd_color = episode["pcd_color"][step_idx] if "pcd_color" in episode else None
    return pcd_xyz, pcd_color


def _save_step_figure(
    rgb: np.ndarray | None,
    depth: np.ndarray | None,
    pcd_xyz: np.ndarray | None,
    pcd_color: np.ndarray | None,
    out_path: Path,
    pcd_max_points: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    if rgb is not None:
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
    else:
        axes[0].text(0.5, 0.5, "RGB missing", ha="center", va="center")
    axes[0].axis("off")

    if depth is not None:
        depth_viz = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        im = axes[1].imshow(depth_viz, cmap="viridis")
        axes[1].set_title("Depth Sensor")
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].text(0.5, 0.5, "Depth missing", ha="center", va="center")
        axes[1].axis("off")

    if pcd_xyz is not None:
        n = pcd_xyz.shape[0]
        if n > pcd_max_points:
            idx = np.random.choice(n, pcd_max_points, replace=False)
            pts = pcd_xyz[idx]
            colors = pcd_color[idx] if pcd_color is not None else None
        else:
            pts = pcd_xyz
            colors = pcd_color

        if colors is not None:
            if colors.dtype != np.float32 and colors.dtype != np.float64:
                colors = colors.astype(np.float32)
            if colors.max() > 1.0:
                colors = colors / 255.0
            axes[2].scatter(pts[:, 0], pts[:, 1], s=1.0, c=colors)
        else:
            axes[2].scatter(pts[:, 0], pts[:, 1], s=1.0)
        axes[2].set_title("Point Cloud (Top View XY)")
        axes[2].set_xlabel("X")
        axes[2].set_ylabel("Y")
        axes[2].set_aspect("equal")
    else:
        axes[2].text(0.5, 0.5, "Point cloud missing", ha="center", va="center")
        axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else data_path / "quick_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="r")
    if args.episode < 0 or args.episode >= replay_buffer.n_episodes:
        raise IndexError(
            f"Episode index {args.episode} out of range, dataset has {replay_buffer.n_episodes} episodes"
        )

    episode = replay_buffer.get_episode(args.episode)
    ep_len = len(episode["robot_state"]) if "robot_state" in episode else len(next(iter(episode.values())))
    start = max(0, args.start_step)
    end = min(ep_len, start + max(1, args.num_steps))

    print(f"Episode keys: {_get_step_keys(episode)}")
    print(f"Episode length: {ep_len}")
    print(f"Exporting steps [{start}, {end}) to: {out_dir}")

    for step_idx in range(start, end):
        rgb = _pick_rgb(episode, step_idx, args.camera_idx)
        depth = _pick_depth(episode, step_idx, args.camera_idx)
        pcd_xyz, pcd_color = _pick_pcd(episode, step_idx)
        out_path = out_dir / f"ep{args.episode:03d}_step{step_idx:04d}.png"
        _save_step_figure(rgb, depth, pcd_xyz, pcd_color, out_path, args.pcd_max_points)

    print("Done.")


if __name__ == "__main__":
    main()
