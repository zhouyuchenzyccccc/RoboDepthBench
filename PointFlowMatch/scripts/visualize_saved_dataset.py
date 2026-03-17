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
    parser.add_argument(
        "--num_steps",
        type=int,
        default=-1,
        help="Number of steps to export. Use -1 to export until episode end.",
    )
    parser.add_argument(
        "--all_cameras",
        action="store_true",
        help="Visualize all cameras for each step (RGB/Depth/Point cloud).",
    )
    parser.add_argument(
        "--camera_idx",
        type=int,
        default=0,
        help="Camera index used in single-camera mode.",
    )
    parser.add_argument(
        "--pcd_max_points",
        type=int,
        default=3000,
        help="Maximum number of point cloud points to draw",
    )
    return parser.parse_args()


CAMERA_NAMES = ["right_shoulder", "left_shoulder", "overhead", "front", "wrist"]


def _get_step_keys(episode: dict[str, np.ndarray]) -> list[str]:
    return sorted(list(episode.keys()))


def _safe_random_downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.random.choice(points.shape[0], max_points, replace=False)
    return points[idx]


def _pick_rgb_list(
    episode: dict[str, np.ndarray], step_idx: int, all_cameras: bool, camera_idx: int
) -> list[tuple[str, np.ndarray | None]]:
    if "images" in episode and episode["images"][step_idx].ndim == 4:
        imgs = episode["images"][step_idx]
        if all_cameras:
            return [(name, imgs[i]) for i, name in enumerate(CAMERA_NAMES)]
        return [(CAMERA_NAMES[camera_idx], imgs[camera_idx])]

    if all_cameras:
        result: list[tuple[str, np.ndarray | None]] = []
        for name in CAMERA_NAMES:
            key = f"rgb_{name}"
            result.append((name, episode[key][step_idx] if key in episode else None))
        return result

    name = CAMERA_NAMES[camera_idx]
    key = f"rgb_{name}"
    return [(name, episode[key][step_idx] if key in episode else None)]


def _pick_depth_list(
    episode: dict[str, np.ndarray], step_idx: int, all_cameras: bool, camera_idx: int
) -> list[tuple[str, np.ndarray | None]]:
    if "depth_sensor" in episode and episode["depth_sensor"][step_idx].ndim == 3:
        depth = episode["depth_sensor"][step_idx]
        if all_cameras:
            return [(name, depth[i]) for i, name in enumerate(CAMERA_NAMES)]
        return [(CAMERA_NAMES[camera_idx], depth[camera_idx])]

    if all_cameras:
        result: list[tuple[str, np.ndarray | None]] = []
        for name in CAMERA_NAMES:
            key = f"depth_sensor_{name}"
            result.append((name, episode[key][step_idx] if key in episode else None))
        return result

    name = CAMERA_NAMES[camera_idx]
    key = f"depth_sensor_{name}"
    return [(name, episode[key][step_idx] if key in episode else None)]


def _pick_camera_point_clouds(
    episode: dict[str, np.ndarray], step_idx: int, all_cameras: bool, camera_idx: int
) -> list[tuple[str, np.ndarray | None]]:
    if all_cameras:
        result: list[tuple[str, np.ndarray | None]] = []
        for name in CAMERA_NAMES:
            key = f"point_cloud_{name}"
            result.append((name, episode[key][step_idx] if key in episode else None))
        return result

    name = CAMERA_NAMES[camera_idx]
    key = f"point_cloud_{name}"
    return [(name, episode[key][step_idx] if key in episode else None)]


def _pick_pcd(episode: dict[str, np.ndarray], step_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    pcd_xyz = episode["pcd_xyz"][step_idx] if "pcd_xyz" in episode else None
    pcd_color = episode["pcd_color"][step_idx] if "pcd_color" in episode else None
    return pcd_xyz, pcd_color


def _save_step_figure(
    rgb_list: list[tuple[str, np.ndarray | None]],
    depth_list: list[tuple[str, np.ndarray | None]],
    camera_pcd_list: list[tuple[str, np.ndarray | None]],
    pcd_xyz: np.ndarray | None,
    pcd_color: np.ndarray | None,
    robot_state: np.ndarray | None,
    out_path: Path,
    pcd_max_points: int,
) -> None:
    import matplotlib.pyplot as plt

    n_cams = max(len(rgb_list), len(depth_list), len(camera_pcd_list))
    n_cols = max(2, n_cams)
    fig, axes = plt.subplots(4, n_cols, figsize=(4.2 * n_cols, 13))

    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for col in range(n_cols):
        axes[0, col].axis("off")
        axes[1, col].axis("off")
        axes[2, col].axis("off")
        axes[3, col].axis("off")

    # Row 1: RGB
    for col, (cam_name, rgb) in enumerate(rgb_list):
        ax = axes[0, col]
        if rgb is None:
            ax.text(0.5, 0.5, f"RGB missing\n{cam_name}", ha="center", va="center")
        else:
            ax.imshow(rgb)
            ax.set_title(f"RGB/{cam_name}")
        ax.axis("off")

    # Row 2: Depth
    for col, (cam_name, depth) in enumerate(depth_list):
        ax = axes[1, col]
        if depth is None:
            ax.text(0.5, 0.5, f"Depth missing\n{cam_name}", ha="center", va="center")
            ax.axis("off")
            continue
        depth_viz = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        im = ax.imshow(depth_viz, cmap="viridis")
        ax.set_title(f"Depth/{cam_name}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 3: Per-camera point clouds (top view)
    for col, (cam_name, cam_pcd) in enumerate(camera_pcd_list):
        ax = axes[2, col]
        if cam_pcd is None or cam_pcd.ndim != 3 or cam_pcd.shape[-1] != 3:
            ax.text(0.5, 0.5, f"PCD missing\n{cam_name}", ha="center", va="center")
            ax.axis("off")
            continue
        pts = cam_pcd.reshape(-1, 3)
        pts = _safe_random_downsample(pts, pcd_max_points)
        ax.scatter(pts[:, 0], pts[:, 1], s=0.8)
        ax.set_title(f"PCD/{cam_name} (XY)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

    # Row 4 col 1: merged pcd
    ax_merged = axes[3, 0]
    if pcd_xyz is None:
        ax_merged.text(0.5, 0.5, "Merged point cloud missing", ha="center", va="center")
        ax_merged.axis("off")
    else:
        if pcd_xyz.shape[0] > pcd_max_points:
            idx = np.random.choice(pcd_xyz.shape[0], pcd_max_points, replace=False)
            pts = pcd_xyz[idx]
            colors = pcd_color[idx] if pcd_color is not None else None
        else:
            pts = pcd_xyz
            colors = pcd_color
        if pcd_color is not None:
            if colors.dtype != np.float32 and colors.dtype != np.float64:
                colors = colors.astype(np.float32)
            if colors.max() > 1.0:
                colors = colors / 255.0
            ax_merged.scatter(pts[:, 0], pts[:, 1], s=1.0, c=colors)
        else:
            ax_merged.scatter(pts[:, 0], pts[:, 1], s=1.0)
        ax_merged.set_title("Merged PCD (XY)")
        ax_merged.set_xlabel("X")
        ax_merged.set_ylabel("Y")
        ax_merged.set_aspect("equal")

    # Row 4 col 2: robot state bar
    if n_cols > 1:
        ax_state = axes[3, 1]
        if robot_state is None:
            ax_state.text(0.5, 0.5, "robot_state missing", ha="center", va="center")
            ax_state.axis("off")
        else:
            idx = np.arange(robot_state.shape[0])
            ax_state.bar(idx, robot_state)
            ax_state.set_title("robot_state")
            ax_state.set_xlabel("dim")
            ax_state.set_ylabel("value")

    fig.suptitle(out_path.stem, fontsize=12)

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
    if args.num_steps < 0:
        end = ep_len
    else:
        end = min(ep_len, start + max(1, args.num_steps))

    print(f"Episode keys: {_get_step_keys(episode)}")
    print(f"Episode length: {ep_len}")
    print(f"Exporting steps [{start}, {end}) to: {out_dir}")

    for step_idx in range(start, end):
        rgb_list = _pick_rgb_list(episode, step_idx, args.all_cameras, args.camera_idx)
        depth_list = _pick_depth_list(episode, step_idx, args.all_cameras, args.camera_idx)
        camera_pcd_list = _pick_camera_point_clouds(
            episode, step_idx, args.all_cameras, args.camera_idx
        )
        pcd_xyz, pcd_color = _pick_pcd(episode, step_idx)
        robot_state = episode["robot_state"][step_idx] if "robot_state" in episode else None
        out_path = out_dir / f"ep{args.episode:03d}_step{step_idx:04d}.png"
        _save_step_figure(
            rgb_list,
            depth_list,
            camera_pcd_list,
            pcd_xyz,
            pcd_color,
            robot_state,
            out_path,
            args.pcd_max_points,
        )

    print("Done.")


if __name__ == "__main__":
    main()
