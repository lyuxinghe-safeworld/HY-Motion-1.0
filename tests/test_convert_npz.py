from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = REPO_ROOT / "scripts" / "convert_npz.py"


def load_converter_module():
    assert CONVERTER_PATH.exists(), f"Expected converter script at {CONVERTER_PATH}"
    spec = importlib.util.spec_from_file_location("local_convert_npz", CONVERTER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_converter_parse_args_accepts_single_file_and_output_dir():
    module = load_converter_module()

    args = module.parse_args(
        [
            "--npz-file",
            "sample.npz",
            "--output-dir",
            "output/motions",
        ]
    )

    assert args.npz_file == "sample.npz"
    assert args.npz_dir is None
    assert args.output_dir == "output/motions"
    assert args.protomotions_root.endswith("ProtoMotions")


def test_resolve_conversion_jobs_emits_motion_output_names(tmp_path: Path):
    module = load_converter_module()

    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    file_a = npz_dir / "walk_000.npz"
    file_b = npz_dir / "turn_000.npz"
    file_a.write_bytes(b"")
    file_b.write_bytes(b"")

    jobs = module.resolve_conversion_jobs(
        npz_file=None,
        npz_dir=str(npz_dir),
        output_dir=str(tmp_path / "motions"),
    )

    assert [(src.name, dst.name) for src, dst in jobs] == [
        ("turn_000.npz", "turn_000.motion"),
        ("walk_000.npz", "walk_000.motion"),
    ]


def test_convert_hymotion_npz_writes_motion_without_dm_control_or_mujoco(
    tmp_path: Path,
):
    module = load_converter_module()
    protomotions_root = Path(module.DEFAULT_PROTOMOTIONS_ROOT)
    assert protomotions_root.exists(), protomotions_root

    npz_path = tmp_path / "jump_000.npz"
    output_path = tmp_path / "jump_000.motion"

    np.savez(
        npz_path,
        poses=np.zeros((3, 72), dtype=np.float32),
        trans=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [0.04, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        mocap_framerate=np.array(30, dtype=np.int32),
    )

    with pytest.warns(
        RuntimeWarning,
        match="No contact candidates found with strict thresholds",
    ):
        module.convert_hymotion_npz(
            npz_path=npz_path,
            output_path=output_path,
            protomotions_root=protomotions_root,
        )

    assert output_path.exists()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        motion = torch.load(output_path, map_location="cpu", weights_only=False)
    assert motion["rigid_body_pos"].shape[0] == 3
    assert motion["local_rigid_body_rot"].shape[-1] == 4
    assert motion["dof_pos"].ndim == 2


def test_contact_detection_falls_back_to_lenient_thresholds_when_strict_finds_none():
    module = load_converter_module()
    positions = torch.zeros((2, 1, 3), dtype=torch.float32)
    velocity = torch.zeros((2, 1, 3), dtype=torch.float32)
    calls: list[tuple[float, float]] = []

    def fake_contact_detector(
        positions: torch.Tensor,
        velocity: torch.Tensor,
        vel_thres: float,
        height_thresh: float,
    ) -> torch.Tensor:
        del positions, velocity
        calls.append((vel_thres, height_thresh))
        if (vel_thres, height_thresh) == (0.15, 0.1):
            return torch.zeros((2, 1), dtype=torch.float32)
        if (vel_thres, height_thresh) == (0.8, 0.15):
            return torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        raise AssertionError(f"Unexpected thresholds: {(vel_thres, height_thresh)}")

    with pytest.warns(RuntimeWarning, match="No contact candidates found"):
        contacts = module._compute_contact_labels_with_fallback(
            positions=positions,
            velocity=velocity,
            compute_contact_labels=fake_contact_detector,
        )

    assert calls == [(0.15, 0.1), (0.8, 0.15)]
    assert contacts.dtype == torch.bool
    assert torch.equal(contacts, torch.tensor([[False], [True]]))


def test_contact_detection_keeps_strict_candidates_without_warning():
    module = load_converter_module()
    positions = torch.zeros((1, 1, 3), dtype=torch.float32)
    velocity = torch.zeros((1, 1, 3), dtype=torch.float32)
    calls: list[tuple[float, float]] = []

    def fake_contact_detector(
        positions: torch.Tensor,
        velocity: torch.Tensor,
        vel_thres: float,
        height_thresh: float,
    ) -> torch.Tensor:
        del positions, velocity
        calls.append((vel_thres, height_thresh))
        return torch.tensor([[1.0]], dtype=torch.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        contacts = module._compute_contact_labels_with_fallback(
            positions=positions,
            velocity=velocity,
            compute_contact_labels=fake_contact_detector,
        )

    assert caught == []
    assert calls == [(0.15, 0.1)]
    assert contacts.dtype == torch.bool
    assert torch.equal(contacts, torch.tensor([[True]]))


def test_converter_main_returns_nonzero_when_any_job_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = load_converter_module()
    npz_path = tmp_path / "broken.npz"
    npz_path.write_bytes(b"placeholder")

    def fail_convert(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "convert_hymotion_npz", fail_convert)

    exit_code = module.main(
        [
            "--npz-file",
            str(npz_path),
            "--output-dir",
            str(tmp_path / "motions"),
        ]
    )

    assert exit_code == 1
