# HY-Motion 1.0 Lite -- Deployment Guide (Low-VRAM GPUs)

This document describes how to deploy **HY-Motion-1.0-Lite** on GPUs with limited VRAM (e.g., NVIDIA L4 with 23 GB) using a Conda environment and 4-bit quantization for the Qwen3-8B text encoder.

## Table of Contents

- [Background and Motivation](#background-and-motivation)
- [Hardware Requirements](#hardware-requirements)
- [Code Modifications](#code-modifications)
- [Step-by-Step Setup](#step-by-step-setup)
  - [1. Create the Conda Environment](#1-create-the-conda-environment)
  - [2. Install PyTorch with CUDA](#2-install-pytorch-with-cuda)
  - [3. Install Project Dependencies](#3-install-project-dependencies)
  - [4. Download Model Weights](#4-download-model-weights)
- [Running Inference](#running-inference)
  - [Single Prompt with Visualization](#single-prompt-with-visualization)
  - [Batch Inference (CLI)](#batch-inference-cli)
  - [Gradio App](#gradio-app)
  - [Custom Prompts](#custom-prompts)
- [Isaac Lab Integration](#isaac-lab-integration)
- [Environment Variables Reference](#environment-variables-reference)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)

---

## Background and Motivation

HY-Motion uses two text encoders during inference:

1. **CLIP ViT-L/14** (`openai/clip-vit-large-patch14`) -- a sentence embedding model (~0.4 GB)
2. **Qwen3-8B** (`Qwen/Qwen3-8B`) -- a large language model used for contextual text encoding (~16 GB in bfloat16)

In the default configuration, both encoders are loaded in bfloat16 alongside the motion DiT model. The combined VRAM usage exceeds what a 24 GB GPU can provide:

| Component | VRAM (bf16) |
|:--|--:|
| Qwen3-8B text encoder | ~16 GB |
| CLIP ViT-L/14 | ~0.4 GB |
| HY-Motion-1.0-Lite DiT (0.46B params) | ~3.5 GB |
| Inference activations & buffers | ~2-4 GB |
| **Total** | **~22-24 GB** |

On a GPU with exactly 23 GB (e.g., NVIDIA L4), the default bf16 loading fails with `torch.OutOfMemoryError` during `pipeline.to(cuda)`.

To solve this, we added **4-bit NormalFloat (NF4) quantization** for the Qwen3-8B text encoder using the `bitsandbytes` library, reducing its VRAM footprint from ~16 GB to ~5 GB. This brings the total under 12 GB, leaving ample headroom for inference.

## Hardware Requirements

| | Minimum | Tested |
|:--|:--|:--|
| GPU | NVIDIA GPU with 16+ GB VRAM and CUDA compute capability >= 7.0 | NVIDIA L4 (23 GB) |
| System RAM | 32 GB | 32 GB |
| Disk | ~20 GB free (model weights + environment) | -- |
| CUDA Driver | >= 12.0 | 12.4 |

## Code Modifications

A single file was modified: `hymotion/network/text_encoders/text_encoder.py`.

### What Changed

The Qwen3-8B loading logic in the `HYTextModel.__init__` method was extended to support optional 4-bit quantization via the `HY_MOTION_LLM_4BIT` environment variable.

**Original code** (lines 101-105):

```python
self.llm_text_encoder = LLM_ENCODER_LAYOUT[llm_type]["text_encoder_class"].from_pretrained(
    LLM_ENCODER_LAYOUT[llm_type]["module_path"],
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
)
```

**Modified code** (lines 101-120):

```python
_load_in_4bit = os.environ.get("HY_MOTION_LLM_4BIT", "0") == "1"
if _load_in_4bit:
    from transformers import BitsAndBytesConfig
    _bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    self.llm_text_encoder = LLM_ENCODER_LAYOUT[llm_type]["text_encoder_class"].from_pretrained(
        LLM_ENCODER_LAYOUT[llm_type]["module_path"],
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        quantization_config=_bnb_config,
    )
else:
    self.llm_text_encoder = LLM_ENCODER_LAYOUT[llm_type]["text_encoder_class"].from_pretrained(
        LLM_ENCODER_LAYOUT[llm_type]["module_path"],
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
```

### How It Works

- When `HY_MOTION_LLM_4BIT=1` is set in the environment, the Qwen3-8B model is loaded with `bitsandbytes` 4-bit NormalFloat quantization (`nf4`). The compute dtype remains `bfloat16`, meaning dequantization happens on-the-fly during matrix multiplications with no change to the forward pass interface.
- When the variable is unset or set to `"0"` (the default), the original bf16 loading path is used. This means the modification is **fully backward-compatible** -- existing setups with sufficient VRAM are unaffected.
- The `bitsandbytes` library (version 0.49.0) is already listed in the project's `requirements.txt`, so no additional dependencies are needed.

### Impact on Output Quality

4-bit quantization introduces a small approximation in the text encoder's hidden states. In practice, the generated motions remain semantically faithful to the prompts. All 57 example prompts from `examples/example_prompts/example_subset.json` ran successfully with this configuration.

---

## Step-by-Step Setup

All commands below assume Miniforge3 (or Miniconda/Anaconda) is installed. Adjust paths as needed.

### 1. Create the Conda Environment

```bash
conda create -n hymotion python=3.10 -y
conda activate hymotion
```

### 2. Install PyTorch with CUDA

Install PyTorch 2.5.1 with CUDA 12.4 support (match your driver version):

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

Verify the installation:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
# Expected: PyTorch 2.5.1+cu124, CUDA available: True
```

For other CUDA versions, consult the [PyTorch installation matrix](https://pytorch.org/get-started/locally/).

### 3. Install Project Dependencies

```bash
cd HY-Motion/
pip install -r requirements.txt
```

This installs all required packages including:
- `transformers==4.53.3` (Hugging Face model loading)
- `bitsandbytes==0.49.0` (4-bit quantization support)
- `accelerate==0.30.1` (efficient model loading)
- `diffusers==0.26.3`, `safetensors==0.5.3`
- `fbxsdkpy==2020.1.post2` (FBX export, optional -- inference works without it)

### 4. Download Model Weights

#### Motion Model (Required)

Download the Lite model (~1.8 GB):

```bash
huggingface-cli download tencent/HY-Motion-1.0 \
    --include "HY-Motion-1.0-Lite/*" \
    --local-dir ckpts/tencent
```

Or the full model (~4 GB, requires >= 26 GB VRAM even with 4-bit text encoder):

```bash
huggingface-cli download tencent/HY-Motion-1.0 \
    --include "HY-Motion-1.0/*" \
    --local-dir ckpts/tencent
```

#### Text Encoders

By default (`USE_HF_MODELS=1`), text encoder weights are downloaded automatically from Hugging Face on first run. To pre-download them for offline use:

```bash
# CLIP (required, ~600 MB)
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ckpts/clip-vit-large-patch14/

# Qwen3-8B (required, ~16 GB download)
huggingface-cli download Qwen/Qwen3-8B --local-dir ckpts/Qwen3-8B
```

If using local checkpoints, set `USE_HF_MODELS=0` so the code reads from `ckpts/` instead of Hugging Face Hub.

#### Expected Directory Structure

```
ckpts/
├── tencent/
│   ├── HY-Motion-1.0-Lite/
│   │   ├── config.yml
│   │   └── latest.ckpt
│   └── HY-Motion-1.0/          # Optional (full model)
│       ├── config.yml
│       └── latest.ckpt
├── clip-vit-large-patch14/      # Only if USE_HF_MODELS=0
└── Qwen3-8B/                    # Only if USE_HF_MODELS=0
```

---

## Running Inference

### Single Prompt (Python CLI)

Generate motion from a single text prompt and save both the NPZ file and a skeleton animation MP4:

```bash
conda activate hymotion
cd ~/code/HY-Motion

HY_MOTION_LLM_4BIT=1 \
USE_HF_MODELS=1 \
python scripts/generate_motion.py \
    --prompt "A person jumps upward with both legs" \
    --duration 3.0 \
    --output-dir output/generated/
```

This produces:
- `output/generated/<slug>/<slug>.npz` -- SMPL-H parameters (identical format to `local_infer.py` output)
- `output/generated/<slug>/<slug>.txt` -- prompt text
- `output/generated/<slug>/<slug>_skeleton.mp4` -- 3D skeleton animation video

**Options:**

| Flag | Default | Description |
|:--|:--|:--|
| `--prompt` | (required) | Text prompt describing the desired motion |
| `--duration` | `3.0` | Motion duration in seconds |
| `--model-path` | `ckpts/tencent/HY-Motion-1.0-Lite` | Path to model directory |
| `--output-dir` | `output/generated` | Output directory |
| `--cfg-scale` | `5.0` | Classifier-free guidance scale |
| `--seed` | `42` | Random seed |
| `--skip-video` | off | Skip skeleton MP4 rendering |
| `--enable-body-model-chunking` | off | Decode the wooden mesh in frame chunks to reduce peak VRAM usage for longer motions |

The NPZ output is directly consumable by `hymotion_isaaclab` for Isaac Lab tracking (see [Isaac Lab Integration](#isaac-lab-integration)).

If `USE_HF_MODELS` is unset, `scripts/generate_motion.py` automatically falls back to `USE_HF_MODELS=1` when local text-encoder checkpoints are missing from `ckpts/`.

### Single Prompt (Bash Wrapper)

Use the wrapper when you want a single command that can either stop at `.npz` or also convert to ProtoMotions `.motion` format:

```bash
conda activate hymotion
cd ~/code/HY-Motion

HY_MOTION_LLM_4BIT=1 \
USE_HF_MODELS=1 \
bash scripts/generate_motion.sh \
    --prompt "A person jumps upward with both legs" \
    --duration 5.0 \
    --enable-body-model-chunking \
    --output-type motion \
    --output-dir output/generated/
```

This creates `output/generated/<slug>/` and writes:
- `<slug>.npz`
- `<slug>.txt`
- `<slug>_skeleton.mp4`
- `<slug>.motion` when `--output-type motion` (default)

**Wrapper options:**

| Flag | Default | Description |
|:--|:--|:--|
| `--prompt` | (required) | Text prompt describing the desired motion |
| `--duration` | `3.0` | Motion duration in seconds |
| `--output-dir` | `output/generated` | Output directory |
| `--output-type` | `motion` | Final artifact to keep: `motion` or `npz` |
| `--enable-body-model-chunking` | off | Reduce peak VRAM usage during wooden-mesh decoding |

### Batch Inference (CLI)

Run the full example prompt set (57 prompts) with 4-bit quantization:

```bash
conda activate hymotion

HY_MOTION_LLM_4BIT=1 \
USE_HF_MODELS=1 \
python3 local_infer.py \
    --model_path ckpts/tencent/HY-Motion-1.0-Lite \
    --num_seeds 1 \
    --disable_rewrite \
    --disable_duration_est
```

**Flag breakdown:**

| Flag | Purpose |
|:--|:--|
| `HY_MOTION_LLM_4BIT=1` | Load Qwen3-8B in 4-bit NF4 quantization (~5 GB instead of ~16 GB) |
| `USE_HF_MODELS=1` | Fetch CLIP and Qwen3-8B from Hugging Face Hub (set to `0` for local weights) |
| `--model_path` | Path to the motion model directory containing `config.yml` and `latest.ckpt` |
| `--num_seeds 1` | Generate 1 motion variant per prompt (reduces VRAM; default is 4) |
| `--disable_rewrite` | Skip LLM-based prompt rewriting (requires separate Text2MotionPrompter model) |
| `--disable_duration_est` | Skip LLM-based duration estimation (uses duration from input file instead) |

**Additional CLI options:**

```bash
# Use specific GPU(s)
--device_ids 0

# Custom input prompts directory
--input_text_dir /path/to/prompts/

# Custom output directory
--output_dir output/my_experiment

# Adjust classifier-free guidance scale (default: 5.0)
--cfg_scale 7.0

# Override denoising steps
--validation_steps 30
```

### Gradio App

To launch the interactive web UI:

```bash
conda activate hymotion

HY_MOTION_LLM_4BIT=1 \
USE_HF_MODELS=1 \
DISABLE_PROMPT_ENGINEERING=True \
python3 gradio_app.py
```

Then open `http://localhost:7860` in your browser.

`DISABLE_PROMPT_ENGINEERING=True` disables the prompt rewrite/duration estimation module, which requires a separate LLM (Text2MotionPrompter) that would consume additional VRAM.

### Custom Prompts

#### Text File Format

Create a `.txt` file with one prompt per line in the format `prompt#duration_frames#id`:

```text
A person walks forward briskly.#120#001
A person performs a jumping jack.#90#002
A person sits down on a chair.#90#003
```

- `duration_frames`: number of frames at 30 FPS (e.g., `90` = 3 seconds). Only used when `--disable_duration_est` is set.
- `id`: optional unique identifier for output file naming.

#### JSON File Format

```json
{
    "walking": [
        "A person walks forward.#120#001",
        "A person walks backward.#120#002"
    ],
    "actions": [
        "A person picks up an object.#90#003"
    ]
}
```

Each top-level key creates a subdirectory in the output folder.

Run with custom prompts:

```bash
HY_MOTION_LLM_4BIT=1 USE_HF_MODELS=1 python3 local_infer.py \
    --model_path ckpts/tencent/HY-Motion-1.0-Lite \
    --input_text_dir /path/to/your/prompts/ \
    --output_dir output/custom_run \
    --num_seeds 1 \
    --disable_rewrite \
    --disable_duration_est
```

---

## Isaac Lab Integration

Generated NPZ files can be converted to ProtoMotions `.motion` format and tracked by a physics-simulated humanoid in Isaac Lab using the [`hymotion_isaaclab`](../hymotion_isaaclab/) package.

### End-to-End Pipeline

```
Text prompt
    │
    ▼
scripts/generate_motion.py (this repo, hymotion conda env)
    │  Outputs: <prompt-slug>/ with .npz, .txt, and skeleton .mp4
    ▼
scripts/convert_npz.py (this repo, ProtoMotions found at ~/code/ProtoMotions or via --protomotions-root)
    │  Converts NPZ -> ProtoMotions .motion format
    ▼
hymotion_isaaclab/scripts/run_tracking.py (env_isaaclab)
    │  Runs ProtoMotions tracker in Isaac Lab
    ▼
Physics simulation of humanoid following the motion
```

### Quick Start

The wrapper below is the shortest path for Isaac Lab: it generates the prompt folder, writes the `.npz`, and converts it to `.motion` in one command.

**Step 1 — Generate `.motion`** (hymotion conda env):

```bash
conda activate hymotion
cd ~/code/HY-Motion

HY_MOTION_LLM_4BIT=1 USE_HF_MODELS=1 bash scripts/generate_motion.sh \
    --prompt "A person walks forward" \
    --output-dir output/generated/
```

This writes:
- `output/generated/a_person_walks_forward/a_person_walks_forward.npz`
- `output/generated/a_person_walks_forward/a_person_walks_forward.txt`
- `output/generated/a_person_walks_forward/a_person_walks_forward_skeleton.mp4`
- `output/generated/a_person_walks_forward/a_person_walks_forward.motion`

**Step 2 — Track in Isaac Lab** (env_isaaclab, requires DISPLAY):

```bash
export DISPLAY=:1
PYTHONPATH="$HOME/code/ProtoMotions:$HOME/code/ProtoMotions/data/scripts:$PYTHONPATH" \
python scripts/run_tracking.py \
    --motion-file ~/code/HY-Motion/output/generated/a_person_walks_forward/a_person_walks_forward.motion \
    --num-envs 1
```

**Manual conversion only** (if you generated `--output-type npz`):

The converter looks for ProtoMotions at `~/code/ProtoMotions` by default. If your checkout lives elsewhere, add `--protomotions-root /path/to/ProtoMotions`.

```bash
cd ~/code/HY-Motion

python scripts/convert_npz.py \
    --npz-file ~/code/HY-Motion/output/generated/a_person_walks_forward/a_person_walks_forward.npz \
    --output-dir output/generated/a_person_walks_forward/
```

### Setup

See the [hymotion_isaaclab README](../hymotion_isaaclab/README.md) for full setup instructions (ProtoMotions checkout, Isaac Lab environment, TurboVNC).

---

## Environment Variables Reference

| Variable | Values | Default | Description |
|:--|:--|:--|:--|
| `HY_MOTION_LLM_4BIT` | `0`, `1` | `0` | Enable 4-bit NF4 quantization for Qwen3-8B text encoder. Set to `1` for GPUs with < 26 GB VRAM. |
| `USE_HF_MODELS` | `0`, `1` | `0` | When `1`, loads CLIP and Qwen3-8B from Hugging Face Hub. When `0`, loads from local `ckpts/` directory. If unset, `scripts/generate_motion.py` falls back to `1` when those local text-encoder checkpoints are missing. |
| `DISABLE_PROMPT_ENGINEERING` | `True`, unset | unset | Disables the Text2MotionPrompter module in the Gradio app (saves VRAM). |
| `HY_MOTION_DEVICE` | `cpu`, unset | unset | Force CPU-only inference (very slow, but works without a GPU). |

---

## Output Format

For the single-prompt scripts (`scripts/generate_motion.py` and `scripts/generate_motion.sh`), each prompt creates its own folder under the output directory:

| File | Description |
|:--|:--|
| `<slug>/<slug>.npz` | NumPy archive containing raw SMPL-H parameters (body pose, hand pose, root translation). |
| `<slug>/<slug>.txt` | The original text prompt associated with this motion. |
| `<slug>/<slug>_skeleton.mp4` | Rendered 3D skeleton animation video. |
| `<slug>/<slug>.motion` | ProtoMotions motion file, only when the bash wrapper runs with `--output-type motion`. |

The single-prompt scripts do not emit the legacy `_000` suffix for single-sample outputs.

For `local_infer.py`, the runtime may additionally write FBX outputs when `fbxsdkpy` is installed, and the batch CLI writes summary files such as:

- `batch_results_YYYYMMDD_HHMMSS.json` -- detailed per-prompt results.
- `batch_summary_YYYYMMDD_HHMMSS.txt` -- aggregate statistics.

---

## Troubleshooting

### `torch.OutOfMemoryError: CUDA out of memory`

- Ensure `HY_MOTION_LLM_4BIT=1` is set.
- Use `--num_seeds 1` to reduce batch size.
- Use the Lite model (`HY-Motion-1.0-Lite`) instead of the full model.
- For `scripts/generate_motion.py` or `bash scripts/generate_motion.sh`, add `--enable-body-model-chunking` before increasing `--duration`.
- Longer durations increase peak VRAM usage during wooden-mesh decoding. `HY-Motion-1.0-Lite` is configured for 360 frames, so durations above 12.0 seconds at 30 FPS are truncated.

### `bitsandbytes` errors

If you see errors related to `bitsandbytes` or `libcuda`, ensure:
- CUDA toolkit version matches PyTorch's CUDA version (12.4 in this setup).
- `bitsandbytes>=0.49.0` is installed: `pip install bitsandbytes==0.49.0`.

### Text encoders fail to download

If Hugging Face downloads stall or fail:
- Install `hf_xet` for faster downloads: `pip install huggingface_hub[hf_xet]`.
- Or pre-download weights manually (see [Download Model Weights](#4-download-model-weights)) and set `USE_HF_MODELS=0`.

### Prompt engineering errors

If you see errors about `prompt_engineering_host` being unavailable, make sure you pass both `--disable_rewrite` and `--disable_duration_est` (or provide a running Text2MotionPrompter service).

### FBX export not working

`fbxsdkpy` only supports Linux with specific `glibc` versions. If FBX export fails, the inference still completes -- motions are saved as `.npz` files and viewable via the HTML visualizer. You can convert NPZ to FBX later on a compatible system.
