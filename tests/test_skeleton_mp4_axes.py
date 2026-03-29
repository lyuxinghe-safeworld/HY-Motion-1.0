from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_RENDERER_PATH = (
    REPO_ROOT / "hymotion" / "utils" / "skeleton_visualization.py"
)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAxes:
    def __init__(self, projection=None):
        self.projection = projection
        self.labels = {}
        self.last_scatter = None
        self.last_view = None
        self.limits = {}
        self.aspect = None
        self.title = None

    def clear(self):
        return None

    def set_title(self, value, *_args, **_kwargs):
        self.title = value

    def set_xlim(self, left, right):
        self.limits["x"] = (left, right)

    def set_ylim(self, bottom, top):
        self.limits["y"] = (bottom, top)

    def set_zlim(self, bottom, top):
        self.limits["z"] = (bottom, top)

    def set_xlabel(self, value):
        self.labels["x"] = value

    def set_ylabel(self, value):
        self.labels["y"] = value

    def set_zlabel(self, value):
        self.labels["z"] = value

    def view_init(self, elev=None, azim=None, roll=None):
        self.last_view = (elev, azim, roll)

    def plot(self, *_args, **_kwargs):
        return None

    def scatter(self, x, y, **_kwargs):
        self.last_scatter = (np.asarray(x), np.asarray(y))

    def plot3D(self, *_args, **_kwargs):
        return None

    def scatter3D(self, x, y, z, **_kwargs):
        self.last_scatter = (np.asarray(x), np.asarray(y), np.asarray(z))

    def set_aspect(self, value, **_kwargs):
        self.aspect = value

    def grid(self, *_args, **_kwargs):
        return None


class FakeFigure:
    def __init__(self):
        self.axes = []

    def add_subplot(self, *_args, **_kwargs):
        axes = FakeAxes(projection=_kwargs.get("projection"))
        self.axes.append(axes)
        return axes


class FakeAnimation:
    def __init__(self, _fig, update, frames, interval):
        self._update = update
        self._last_frame = frames - 1 if isinstance(frames, int) else 0
        self.interval = interval

    def save(self, *_args, **_kwargs):
        self._update(self._last_frame)


def build_sample_keypoints() -> np.ndarray:
    xyz = np.zeros((2, 22, 3), dtype=np.float32)
    xyz[:, :, 0] = np.linspace(-0.5, 0.5, 22, dtype=np.float32)
    xyz[:, :, 1] = 1.0
    xyz[1, :, 1] = 1.25
    xyz[1, :, 2] = 2.5
    return xyz


def test_scripts_reuse_shared_renderer():
    assert SHARED_RENDERER_PATH.exists()

    shared_module = importlib.import_module("hymotion.utils.skeleton_visualization")
    visualize_module = load_module(
        "visualize_skeleton_module_shared_check",
        REPO_ROOT / "scripts" / "visualize_skeleton.py",
    )
    generate_module = load_module(
        "generate_motion_module_shared_check",
        REPO_ROOT / "scripts" / "generate_motion.py",
    )

    assert visualize_module.render_skeleton_mp4 is shared_module.render_skeleton_mp4
    assert generate_module.render_skeleton_mp4 is shared_module.render_skeleton_mp4


def test_shared_renderer_draws_side_front_and_3d_views(
    monkeypatch,
    tmp_path: Path,
):
    module = importlib.import_module("hymotion.utils.skeleton_visualization")
    fake_figure = FakeFigure()

    monkeypatch.setattr(module.plt, "figure", lambda *args, **kwargs: fake_figure)
    monkeypatch.setattr(module.plt, "close", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "FuncAnimation", FakeAnimation)
    monkeypatch.setattr(module, "FFMpegWriter", lambda *args, **kwargs: object())

    xyz = build_sample_keypoints()
    module.render_skeleton_mp4(xyz, str(tmp_path / "out.mp4"), "walk", fps=30)

    assert len(fake_figure.axes) == 3
    side_axes, front_axes, view3d_axes = fake_figure.axes

    side_x, side_y = side_axes.last_scatter
    front_x, front_y = front_axes.last_scatter
    _, plotted_y, plotted_z = view3d_axes.last_scatter

    np.testing.assert_allclose(side_x, xyz[1, :, 2])
    np.testing.assert_allclose(side_y, xyz[1, :, 1])
    np.testing.assert_allclose(front_x, xyz[1, :, 0])
    np.testing.assert_allclose(front_y, xyz[1, :, 1])
    np.testing.assert_allclose(plotted_y, xyz[1, :, 2])
    np.testing.assert_allclose(plotted_z, xyz[1, :, 1])

    assert side_axes.limits["y"][0] == 0
    assert front_axes.limits["y"][0] == 0
    assert view3d_axes.labels == {"x": "X", "y": "Forward (Z)", "z": "Up (Y)"}
    assert view3d_axes.last_view == (25, 45, 0)
