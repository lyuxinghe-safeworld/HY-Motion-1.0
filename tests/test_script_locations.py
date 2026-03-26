from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_entrypoint_scripts_live_under_scripts_directory():
    assert (REPO_ROOT / "scripts" / "generate_motion.py").exists()
    assert (REPO_ROOT / "scripts" / "visualize_skeleton.py").exists()
    assert not (REPO_ROOT / "generate_motion.py").exists()
    assert not (REPO_ROOT / "visualize_skeleton.py").exists()
