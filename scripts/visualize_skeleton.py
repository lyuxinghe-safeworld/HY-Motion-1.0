#!/usr/bin/env python3
"""Generate a 3D skeleton animation MP4 from an HY-Motion text prompt.

Produces the same visual style as closd_isaaclab/scripts/verify_diffusion.py:
  - matplotlib 3D scatter + line skeleton
  - 600x600 resolution, colored kinematic chains, black joint dots
  - title with prompt text and frame counter

Usage:
    # With conda environment:
    conda activate hymotion
    HY_MOTION_LLM_4BIT=1 USE_HF_MODELS=1 python scripts/visualize_skeleton.py \
        --prompt "A person walks forward" --duration 4.0

    # With local model weights:
    HY_MOTION_LLM_4BIT=1 USE_HF_MODELS=0 python scripts/visualize_skeleton.py \
        --prompt "A person jumps upward" --duration 3.0 \
        --model-path ckpts/tencent/HY-Motion-1.0-Lite
"""

import argparse
import json
import re
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MP4_AXIS_LABELS = ("X", "Forward (Z)", "Up (Y)")
MP4_CAMERA_ELEV = 25
MP4_CAMERA_AZIM = 45


def remap_hymotion_xyz_for_matplotlib(xyz: np.ndarray) -> np.ndarray:
    """Map HY-Motion y-up coordinates into matplotlib's z-up display space."""
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected keypoints shaped (T, J, 3), got {xyz.shape}")

    return xyz[..., [0, 2, 1]]


def apply_mp4_camera_view(ax) -> None:
    """Set a front-facing default view for saved skeleton MP4s."""
    ax.view_init(elev=MP4_CAMERA_ELEV, azim=MP4_CAMERA_AZIM, roll=0)


def resolve_repo_input_path(path: str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute() or path_obj.exists():
        return str(path_obj)

    return str(REPO_ROOT / path_obj)


# ---------------------------------------------------------------------------
# SMPL-H 52-joint skeleton: kinematic chains for visualization
# (indices into the 52-joint ordering used by WoodenMesh / HY-Motion)
# ---------------------------------------------------------------------------
#  0: Pelvis        1: L_Hip       2: R_Hip       3: Spine1      4: L_Knee
#  5: R_Knee        6: Spine2      7: L_Ankle     8: R_Ankle     9: Spine3
# 10: L_Foot       11: R_Foot     12: Neck       13: L_Collar   14: R_Collar
# 15: Head         16: L_Shoulder 17: R_Shoulder 18: L_Elbow    19: R_Elbow
# 20: L_Wrist      21: R_Wrist    22-36: L_Hand  37-51: R_Hand

KINEMATIC_CHAINS = [
    [0, 1, 4, 7, 10],              # left leg
    [0, 2, 5, 8, 11],              # right leg
    [0, 3, 6, 9, 12, 15],          # spine + head
    [9, 13, 16, 18, 20],           # left arm
    [9, 14, 17, 19, 21],           # right arm
]
CHAIN_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#e67e22"]

# Body joints to render (skip finger joints 22-51 for cleaner visualization)
BODY_JOINT_INDICES = list(range(22))


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text[:80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 3D skeleton animation MP4 from an HY-Motion text prompt.",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt describing the desired motion (required unless --npz is used).",
    )
    parser.add_argument(
        "--duration", type=float, default=3.0,
        help="Motion duration in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--model-path", type=str, default="ckpts/tencent/HY-Motion-1.0-Lite",
        help="Path to HY-Motion model directory (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/skeleton_viz",
        help="Directory for output files (default: %(default)s).",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=5.0,
        help="Classifier-free guidance scale (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: %(default)s).",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Output video FPS (default: %(default)s, matches HY-Motion output).",
    )
    parser.add_argument(
        "--npz", type=str, default=None,
        help="Skip generation; render from an existing .npz file instead.",
    )
    return parser.parse_args()


def generate_motion(prompt: str, duration: float, model_path: str,
                    cfg_scale: float, seed: int) -> dict:
    """Run HY-Motion inference and return the raw NPZ-style dict."""
    import os.path as osp
    from hymotion.utils.t2m_runtime import T2MRuntime

    model_path = resolve_repo_input_path(model_path)
    cfg = osp.join(model_path, "config.yml")
    ckpt = osp.join(model_path, "latest.ckpt")
    if not os.path.exists(cfg):
        raise FileNotFoundError(f"Config not found: {cfg}")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"Loading model from {model_path} ...")
    runtime = T2MRuntime(
        config_path=cfg,
        ckpt_name=ckpt,
        disable_prompt_engineering=True,
    )

    print(f"Generating motion for: '{prompt}' ({duration:.1f}s, seed={seed})")
    _, _, model_output = runtime.generate_motion(
        text=prompt,
        seeds_csv=str(seed),
        duration=duration,
        cfg_scale=cfg_scale,
        output_format="dict",
    )
    return model_output


def extract_keypoints(model_output: dict) -> np.ndarray:
    """Extract 3D joint positions from model output.

    The pipeline's keypoints3d are in pelvis-local space (WoodenMesh.forward
    returns posed_joints without root translation). We add transl back to get
    world-space positions.

    Returns:
        keypoints: (T, 52, 3) numpy array of joint positions.
    """
    # keypoints3d from the pipeline is pelvis-local (no root translation)
    kp = model_output["keypoints3d"]    # (B, T, J, 3)
    transl = model_output["transl"]     # (B, T, 3)

    # Take the first sample
    if kp.dim() == 4:
        kp = kp[0]          # (T, J, 3)
        transl = transl[0]  # (T, 3)

    # Add root translation to get world-space positions
    kp = kp + transl.unsqueeze(1)  # (T, J, 3) + (T, 1, 3)

    # Ground alignment: shift so minimum y = 0
    min_y = kp[..., 1].min()
    kp[..., 1] -= min_y

    return kp.cpu().numpy()


def load_npz_keypoints(npz_path: str) -> tuple:
    """Load an HY-Motion .npz file and compute joint positions.

    Returns:
        (keypoints, prompt): keypoints is (T, 52, 3) numpy array.
    """
    from hymotion.pipeline.body_model import WoodenMesh

    f = np.load(npz_path, allow_pickle=True)
    poses = torch.from_numpy(f["poses"]).float()   # (T, 156)
    trans = torch.from_numpy(f["trans"]).float()    # (T, 3)

    model = WoodenMesh(
        str(REPO_ROOT / "scripts" / "gradio" / "static" / "assets" / "dump_wooden")
    )
    with torch.no_grad():
        out = model.forward({"poses": poses, "trans": trans})
    # WoodenMesh.forward returns keypoints3d WITHOUT root translation,
    # so we add it back to get world-space positions.
    kp = out["keypoints3d"] + trans.unsqueeze(1)  # (T, J, 3) + (T, 1, 3)
    kp = kp.cpu().numpy()

    # Ground alignment
    kp[..., 1] -= kp[..., 1].min()

    # Try to read prompt from companion .txt file
    txt_path = npz_path.replace(".npz", ".txt")
    prompt = ""
    if os.path.exists(txt_path):
        with open(txt_path) as fh:
            prompt = fh.read().strip()

    return kp, prompt


def render_skeleton_mp4(xyz: np.ndarray, mp4_path: str, title: str,
                        fps: int = 30) -> None:
    """Render a 3D skeleton animation to MP4.

    Matches the visual style of closd_isaaclab verify_diffusion.py:
      - 600x600 figure, 3D projection
      - Colored kinematic chains with linewidth=2
      - Black scatter joints with s=10
      - Title with prompt + frame counter
      - Fixed axis limits from data extent

    Args:
        xyz: (T, J, 3) joint positions.
        mp4_path: Output file path.
        title: Prompt text shown as title.
        fps: Frame rate.
    """
    # Use only body joints for visualization (skip fingers)
    body_xyz = xyz[:, BODY_JOINT_INDICES, :]
    plot_xyz = remap_hymotion_xyz_for_matplotlib(body_xyz)
    T = plot_xyz.shape[0]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Compute fixed axis limits from full trajectory
    mins = plot_xyz.min(axis=(0, 1))
    maxs = plot_xyz.max(axis=(0, 1))
    center = (mins + maxs) / 2
    span = (maxs - mins).max() * 0.6

    def update(frame):
        ax.clear()
        ax.set_title(f"{title}\nFrame {frame}/{T}", fontsize=10)
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_xlabel(MP4_AXIS_LABELS[0])
        ax.set_ylabel(MP4_AXIS_LABELS[1])
        ax.set_zlabel(MP4_AXIS_LABELS[2])
        apply_mp4_camera_view(ax)

        pts = plot_xyz[frame]
        for chain, color in zip(KINEMATIC_CHAINS, CHAIN_COLORS):
            chain_pts = pts[chain]
            ax.plot3D(
                chain_pts[:, 0], chain_pts[:, 1], chain_pts[:, 2],
                color=color, linewidth=2,
            )
        ax.scatter3D(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=10, c="black", zorder=5,
        )

    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps)
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    ani.save(str(mp4_path), writer=writer)
    plt.close(fig)
    print(f"Saved: {mp4_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.npz and not args.prompt:
        print("Error: --prompt is required when not using --npz")
        raise SystemExit(1)

    if args.npz:
        # --npz mode: render from existing file
        print(f"Loading from NPZ: {args.npz}")
        kp, prompt = load_npz_keypoints(args.npz)
        if not prompt:
            prompt = Path(args.npz).stem
        fps = args.fps
    else:
        # Generation mode
        prompt = args.prompt
        model_output = generate_motion(
            prompt=prompt,
            duration=args.duration,
            model_path=args.model_path,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
        )
        kp = extract_keypoints(model_output)
        fps = args.fps

    T = kp.shape[0]
    print(f"Motion: {T} frames, {T / fps:.2f}s @ {fps} FPS")

    slug = slugify(prompt)
    mp4_path = output_dir / f"{slug}_skeleton.mp4"

    print(f"Rendering {mp4_path} ...")
    render_skeleton_mp4(kp, str(mp4_path), prompt, fps=fps)

    # Save keypoints for downstream use
    kp_path = output_dir / f"{slug}_keypoints.npy"
    np.save(str(kp_path), kp)

    print()
    print("=" * 60)
    print("Done")
    print(f"  Prompt    : {prompt}")
    print(f"  Frames    : {T}")
    print(f"  FPS       : {fps}")
    print(f"  Duration  : {T / fps:.2f}s")
    print(f"  Video     : {mp4_path.resolve()}")
    print(f"  Keypoints : {kp_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
