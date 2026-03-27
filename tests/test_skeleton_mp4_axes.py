from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAxes:
    def __init__(self):
        self.labels = {}
        self.last_scatter = None
        self.last_view = None

    def clear(self):
        return None

    def set_title(self, *_args, **_kwargs):
        return None

    def set_xlim(self, *_args, **_kwargs):
        return None

    def set_ylim(self, *_args, **_kwargs):
        return None

    def set_zlim(self, *_args, **_kwargs):
        return None

    def set_xlabel(self, value):
        self.labels["x"] = value

    def set_ylabel(self, value):
        self.labels["y"] = value

    def set_zlabel(self, value):
        self.labels["z"] = value

    def view_init(self, elev=None, azim=None, roll=None):
        self.last_view = (elev, azim, roll)

    def plot3D(self, *_args, **_kwargs):
        return None

    def scatter3D(self, x, y, z, **_kwargs):
        self.last_scatter = (np.asarray(x), np.asarray(y), np.asarray(z))


class FakeFigure:
    def __init__(self, axes: FakeAxes):
        self._axes = axes

    def add_subplot(self, *_args, **_kwargs):
        return self._axes


class FakeAnimation:
    def __init__(self, _fig, update, frames, interval):
        self._update = update
        self._last_frame = frames - 1 if isinstance(frames, int) else 0
        self.interval = interval

    def save(self, *_args, **_kwargs):
        self._update(self._last_frame)


def build_sample_keypoints() -> np.ndarray:
    xyz = np.zeros((2, 22, 3), dtype=np.float32)
    xyz[:, :, 1] = 1.0
    xyz[1, :, 1] = 1.25
    xyz[1, :, 2] = 2.5
    return xyz


@pytest.mark.parametrize(
    ("module_name", "module_path"),
    [
        ("visualize_skeleton_module", REPO_ROOT / "scripts" / "visualize_skeleton.py"),
        ("generate_motion_module", REPO_ROOT / "scripts" / "generate_motion.py"),
    ],
)
def test_skeleton_mp4_renderers_use_y_up_and_z_forward(
    monkeypatch,
    tmp_path: Path,
    module_name: str,
    module_path: Path,
):
    module = load_module(
        module_name,
        module_path,
    )
    fake_axes = FakeAxes()

    if hasattr(module, "plt"):
        monkeypatch.setattr(module.plt, "figure", lambda *args, **kwargs: FakeFigure(fake_axes))
        monkeypatch.setattr(module.plt, "close", lambda *_args, **_kwargs: None)
    else:
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "figure", lambda *args, **kwargs: FakeFigure(fake_axes))
        monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    if hasattr(module, "FuncAnimation"):
        monkeypatch.setattr(module, "FuncAnimation", FakeAnimation)
    else:
        import matplotlib.animation as animation

        monkeypatch.setattr(animation, "FuncAnimation", FakeAnimation)

    if hasattr(module, "FFMpegWriter"):
        monkeypatch.setattr(module, "FFMpegWriter", lambda *args, **kwargs: object())
    else:
        import matplotlib.animation as animation

        monkeypatch.setattr(animation, "FFMpegWriter", lambda *args, **kwargs: object())

    xyz = build_sample_keypoints()
    module.render_skeleton_mp4(xyz, str(tmp_path / "out.mp4"), "walk", fps=30)

    _, plotted_y, plotted_z = fake_axes.last_scatter
    np.testing.assert_allclose(plotted_y, xyz[1, :, 2])
    np.testing.assert_allclose(plotted_z, xyz[1, :, 1])
    assert fake_axes.labels == {"x": "X", "y": "Forward (Z)", "z": "Up (Y)"}
    assert fake_axes.last_view == (25, 45, 0)
