from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


MP4_AXIS_LABELS = ("X", "Forward (Z)", "Up (Y)")
MP4_CAMERA_ELEV = 25
MP4_CAMERA_AZIM = 45

KINEMATIC_CHAINS = [
    [0, 1, 4, 7, 10],
    [0, 2, 5, 8, 11],
    [0, 3, 6, 9, 12, 15],
    [9, 13, 16, 18, 20],
    [9, 14, 17, 19, 21],
]
CHAIN_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#e67e22"]
BODY_JOINT_INDICES = list(range(22))
SIDE_AXIS_LABELS = ("Forward (Z)", "Up (Y)")
FRONT_AXIS_LABELS = ("X", "Up (Y)")


def remap_hymotion_xyz_for_matplotlib(xyz: np.ndarray) -> np.ndarray:
    """Map HY-Motion y-up coordinates into matplotlib's z-up display space."""
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected keypoints shaped (T, J, 3), got {xyz.shape}")

    return xyz[..., [0, 2, 1]]


def apply_mp4_camera_view(ax) -> None:
    """Set a front-facing default view for saved skeleton MP4s."""
    ax.view_init(elev=MP4_CAMERA_ELEV, azim=MP4_CAMERA_AZIM, roll=0)


def _compute_axis_limits(values: np.ndarray, floor: float | None = None) -> tuple[float, float]:
    minimum = float(values.min())
    maximum = float(values.max())
    span = maximum - minimum
    padding = max(span * 0.1, 1e-3)
    lower = minimum - padding
    upper = maximum + padding

    if floor is not None:
        lower = floor

    if lower == upper:
        upper = lower + 1.0

    return lower, upper


def _draw_projected_panel(
    ax,
    x_values: np.ndarray,
    y_values: np.ndarray,
    frame: int,
    title: str,
    x_label: str,
    y_label: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    ax.clear()
    ax.set_title(title, fontsize=10)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    frame_x = x_values[frame]
    frame_y = y_values[frame]
    for chain, color in zip(KINEMATIC_CHAINS, CHAIN_COLORS):
        ax.plot(frame_x[chain], frame_y[chain], color=color, linewidth=2)
    ax.scatter(frame_x, frame_y, s=10, c="black", zorder=5)


def render_skeleton_mp4(
    xyz: np.ndarray,
    mp4_path: str,
    title: str,
    fps: int = 30,
) -> None:
    """Render a multiview skeleton animation to MP4."""
    body_xyz = xyz[:, BODY_JOINT_INDICES, :]
    plot_xyz = remap_hymotion_xyz_for_matplotlib(body_xyz)
    side_x = body_xyz[..., 2]
    side_y = body_xyz[..., 1]
    front_x = body_xyz[..., 0]
    front_y = body_xyz[..., 1]
    num_frames = plot_xyz.shape[0]

    fig = plt.figure(figsize=(10, 10))
    side_ax = fig.add_subplot(2, 2, 1)
    front_ax = fig.add_subplot(2, 2, 2)
    view3d_ax = fig.add_subplot(2, 1, 2, projection="3d")

    mins = plot_xyz.min(axis=(0, 1))
    maxs = plot_xyz.max(axis=(0, 1))
    center = (mins + maxs) / 2
    span = (maxs - mins).max() * 0.6
    side_x_limits = _compute_axis_limits(side_x)
    front_x_limits = _compute_axis_limits(front_x)
    height_limits = _compute_axis_limits(body_xyz[..., 1], floor=0.0)

    def update(frame: int) -> None:
        _draw_projected_panel(
            side_ax,
            side_x,
            side_y,
            frame,
            f"Side View\nFrame {frame}/{num_frames}",
            SIDE_AXIS_LABELS[0],
            SIDE_AXIS_LABELS[1],
            side_x_limits,
            height_limits,
        )
        _draw_projected_panel(
            front_ax,
            front_x,
            front_y,
            frame,
            f"Front View\nFrame {frame}/{num_frames}",
            FRONT_AXIS_LABELS[0],
            FRONT_AXIS_LABELS[1],
            front_x_limits,
            height_limits,
        )

        view3d_ax.clear()
        view3d_ax.set_title(f"{title}\nFrame {frame}/{num_frames}", fontsize=10)
        view3d_ax.set_xlim(center[0] - span, center[0] + span)
        view3d_ax.set_ylim(center[1] - span, center[1] + span)
        view3d_ax.set_zlim(center[2] - span, center[2] + span)
        view3d_ax.set_xlabel(MP4_AXIS_LABELS[0])
        view3d_ax.set_ylabel(MP4_AXIS_LABELS[1])
        view3d_ax.set_zlabel(MP4_AXIS_LABELS[2])
        apply_mp4_camera_view(view3d_ax)

        pts = plot_xyz[frame]
        for chain, color in zip(KINEMATIC_CHAINS, CHAIN_COLORS):
            chain_pts = pts[chain]
            view3d_ax.plot3D(
                chain_pts[:, 0], chain_pts[:, 1], chain_pts[:, 2],
                color=color, linewidth=2,
            )
        view3d_ax.scatter3D(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=10, c="black", zorder=5,
        )

    ani = FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps)
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    ani.save(str(mp4_path), writer=writer)
    plt.close(fig)
    print(f"Saved: {mp4_path}")
