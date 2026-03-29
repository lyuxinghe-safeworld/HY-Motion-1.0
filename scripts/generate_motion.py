#!/usr/bin/env python3
"""Generate human motion from a text prompt, save outputs under a prompt folder.

Creates `output-dir/<prompt-slug>/` and stores the generated NPZ, prompt text,
and skeleton animation there. The NPZ files are identical to `local_infer.py`
output and are consumable by this repo's `scripts/convert_npz.py`.

Usage:
    conda activate hymotion
    cd ~/code/HY-Motion

    # Generate motion + skeleton video under output/generated/<prompt-slug>/
    HY_MOTION_LLM_4BIT=1 python scripts/generate_motion.py \
        --prompt "A person walks forward" \
        --enable-body-model-chunking \
        --output-dir output/generated/

    # Then convert to .motion from this repo:
    python scripts/convert_npz.py \
        --npz-file output/generated/a_person_walks_forward/a_person_walks_forward.npz \
        --output-dir output/generated/a_person_walks_forward/
"""

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hymotion.utils.skeleton_visualization import render_skeleton_mp4


def resolve_repo_input_path(path: str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute() or path_obj.exists():
        return str(path_obj)

    return str(REPO_ROOT / path_obj)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text[:80]


def get_prompt_output_dir(output_dir: str | Path, prompt: str) -> Path:
    return Path(output_dir) / slugify(prompt)


def get_sample_filename(base_filename: str, sample_index: int, batch_size: int) -> str:
    if batch_size == 1:
        return f"{base_filename}.npz"
    return f"{base_filename}_{sample_index:03d}.npz"


def get_pipeline_arg_overrides(
    enable_body_model_chunking: bool,
) -> dict[str, int]:
    if not enable_body_model_chunking:
        return {}
    return {"body_model_chunk_frames": 32}


def configure_text_encoder_environment(
    repo_root: str | Path = REPO_ROOT,
    environ: dict[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ
    repo_root = Path(repo_root)
    local_clip_path = repo_root / "ckpts" / "clip-vit-large-patch14"
    local_qwen_path = repo_root / "ckpts" / "Qwen3-8B"
    missing_paths = [
        path for path in (local_clip_path, local_qwen_path) if not path.is_dir()
    ]

    use_hf_models = env.get("USE_HF_MODELS")
    if use_hf_models == "1":
        return "1"

    if use_hf_models is None:
        if missing_paths:
            env["USE_HF_MODELS"] = "1"
            return "1"

        env["USE_HF_MODELS"] = "0"
        return "0"

    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "USE_HF_MODELS=0 requires local text encoder checkpoints at "
            f"{missing_text}. Set USE_HF_MODELS=1 to download them from "
            "Hugging Face instead."
        )

    return "0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate motion from a text prompt, save NPZ + skeleton MP4.",
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt describing the desired motion.",
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
        "--output-dir", type=str, default="output/generated",
        help="Output directory (default: %(default)s).",
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
        help="Output FPS (default: %(default)s, matches HY-Motion).",
    )
    parser.add_argument(
        "--skip-video", action="store_true",
        help="Skip skeleton MP4 rendering.",
    )
    parser.add_argument(
        "--enable-body-model-chunking",
        action="store_true",
        help="Decode wooden mesh in frame chunks to reduce peak VRAM usage.",
    )
    return parser.parse_args()


def generate_motion(
    prompt,
    duration,
    model_path,
    cfg_scale,
    seed,
    enable_body_model_chunking=False,
):
    """Run HY-Motion inference, return model_output dict."""
    import os.path as osp

    previous_use_hf_models = os.environ.get("USE_HF_MODELS")
    selected_use_hf_models = configure_text_encoder_environment()
    if previous_use_hf_models is None and selected_use_hf_models == "1":
        print(
            "USE_HF_MODELS not set and local text encoder checkpoints were "
            "not found. Falling back to Hugging Face downloads."
        )

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
        pipeline_arg_overrides=get_pipeline_arg_overrides(
            enable_body_model_chunking
        ),
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


def save_npz(model_output, output_dir, base_filename):
    """Save NPZ in the same format as save_visualization_data / local_infer.py.

    Uses construct_smpl_data_dict to ensure identical output. Single-sample
    outputs omit the legacy `_000` suffix.
    """
    import numpy as np
    import torch

    from hymotion.pipeline.body_model import construct_smpl_data_dict

    rot6d = model_output["rot6d"]    # (B, T, J, 6)
    transl = model_output["transl"]  # (B, T, 3)
    batch_size = rot6d.shape[0]

    npz_paths = []
    for bb in range(batch_size):
        smpl_data = construct_smpl_data_dict(rot6d[bb].clone(), transl[bb].clone())

        npz_dict = {}
        npz_dict["gender"] = np.array([smpl_data.get("gender", "neutral")], dtype=str)
        for key in ["Rh", "trans", "poses", "betas"]:
            if key in smpl_data:
                val = smpl_data[key]
                if isinstance(val, (list, tuple)):
                    val = np.array(val)
                elif isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                npz_dict[key] = val

        sample_filename = get_sample_filename(base_filename, bb, batch_size)
        sample_path = os.path.join(output_dir, sample_filename)
        np.savez_compressed(sample_path, **npz_dict)
        npz_paths.append(sample_path)
        print(f"  Saved NPZ: {sample_path}")

    return npz_paths


def extract_keypoints(model_output):
    """Extract 3D joint positions from model output (world-space, ground-aligned)."""
    kp = model_output["keypoints3d"]  # (B, T, J, 3)
    transl = model_output["transl"]   # (B, T, 3)

    if kp.dim() == 4:
        kp = kp[0]
        transl = transl[0]

    # keypoints3d is pelvis-local; add root translation for world-space
    kp = kp + transl.unsqueeze(1)

    # Ground alignment: shift so minimum y = 0
    min_y = kp[..., 1].min()
    kp[..., 1] -= min_y

    return kp.cpu().numpy()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify(args.prompt)
    prompt_output_dir = get_prompt_output_dir(output_dir, args.prompt)
    prompt_output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate motion
    model_output = generate_motion(
        prompt=args.prompt,
        duration=args.duration,
        model_path=args.model_path,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        enable_body_model_chunking=args.enable_body_model_chunking,
    )

    # Step 2: Save NPZ (identical to local_infer.py output)
    print("Saving NPZ files...")
    npz_paths = save_npz(model_output, str(prompt_output_dir), slug)

    # Step 3: Save prompt text alongside NPZ
    for npz_path in npz_paths:
        txt_path = npz_path.replace(".npz", ".txt")
        with open(txt_path, "w") as f:
            f.write(args.prompt)

    # Step 4: Render skeleton visualization
    if not args.skip_video:
        print("Rendering skeleton visualization...")
        kp = extract_keypoints(model_output)
        T = kp.shape[0]
        mp4_path = prompt_output_dir / f"{slug}_skeleton.mp4"
        render_skeleton_mp4(kp, str(mp4_path), args.prompt, fps=args.fps)
    else:
        T = model_output["rot6d"].shape[1]

    # Summary
    print()
    print("=" * 60)
    print("Done")
    print(f"  Prompt   : {args.prompt}")
    print(f"  Output   : {prompt_output_dir}")
    print(f"  Frames   : {T}")
    print(f"  Duration : {T / args.fps:.2f}s @ {args.fps} FPS")
    for p in npz_paths:
        print(f"  NPZ      : {p}")
    if not args.skip_video:
        print(f"  Video    : {mp4_path}")
    print()
    print("Next step — convert to .motion from this repo:")
    print(
        f"  python scripts/convert_npz.py --npz-file {npz_paths[0]} "
        f"--output-dir {prompt_output_dir}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
