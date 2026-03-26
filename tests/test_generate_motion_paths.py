from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATE_MOTION_PATH = REPO_ROOT / "scripts" / "generate_motion.py"


def load_generate_motion_module():
    spec = importlib.util.spec_from_file_location(
        "generate_motion_module",
        GENERATE_MOTION_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_get_prompt_output_dir_uses_prompt_slug(tmp_path: Path):
    module = load_generate_motion_module()

    prompt_dir = module.get_prompt_output_dir(
        tmp_path / "generated",
        "A person walks forward",
    )

    assert prompt_dir == tmp_path / "generated" / "a_person_walks_forward"


def test_save_npz_omits_index_suffix_for_single_sample(tmp_path: Path):
    module = load_generate_motion_module()
    model_output = {
        "rot6d": torch.zeros((1, 2, 22, 6), dtype=torch.float32),
        "transl": torch.zeros((1, 2, 3), dtype=torch.float32),
    }

    npz_paths = module.save_npz(
        model_output=model_output,
        output_dir=str(tmp_path),
        base_filename="a_person_walks_forward",
    )

    assert npz_paths == [str(tmp_path / "a_person_walks_forward.npz")]
    assert (tmp_path / "a_person_walks_forward.npz").exists()


def test_get_pipeline_arg_overrides_only_enables_chunking_when_requested():
    module = load_generate_motion_module()

    assert module.get_pipeline_arg_overrides(False) == {}
    assert module.get_pipeline_arg_overrides(True) == {
        "body_model_chunk_frames": 32,
    }


def test_configure_text_encoder_environment_defaults_to_hf_when_local_ckpts_missing(
    tmp_path: Path,
):
    module = load_generate_motion_module()
    env = {}

    selected_mode = module.configure_text_encoder_environment(
        repo_root=tmp_path,
        environ=env,
    )

    assert selected_mode == "1"
    assert env["USE_HF_MODELS"] == "1"


def test_configure_text_encoder_environment_rejects_explicit_local_mode_without_ckpts(
    tmp_path: Path,
):
    module = load_generate_motion_module()
    env = {"USE_HF_MODELS": "0"}

    try:
        module.configure_text_encoder_environment(
            repo_root=tmp_path,
            environ=env,
        )
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing local checkpoints")

    assert "USE_HF_MODELS=0" in message
    assert "ckpts/clip-vit-large-patch14" in message
    assert "ckpts/Qwen3-8B" in message
    assert "USE_HF_MODELS=1" in message
