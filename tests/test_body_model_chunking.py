from __future__ import annotations

import torch

from hymotion.pipeline import body_model


class DummyBodyModel:
    def __init__(self):
        self.calls = []

    def forward(self, params):
        frames = params["rot6d"].shape[0]
        self.calls.append(frames)
        return {
            "vertices": torch.zeros((frames, 2, 3), dtype=torch.float32),
            "vertices_wotrans": torch.zeros((frames, 2, 3), dtype=torch.float32),
            "keypoints3d": torch.zeros((frames, 5, 3), dtype=torch.float32),
        }


def test_forward_params_in_chunks_splits_long_sequences():
    model = DummyBodyModel()
    params = {
        "rot6d": torch.zeros((150, 22, 6), dtype=torch.float32),
        "trans": torch.zeros((150, 3), dtype=torch.float32),
    }

    output = body_model.forward_params_in_chunks(
        model,
        params=params,
        chunk_size=32,
    )

    assert model.calls == [32, 32, 32, 32, 22]
    assert output["vertices"].shape == (150, 2, 3)
    assert output["vertices_wotrans"].shape == (150, 2, 3)
    assert output["keypoints3d"].shape == (150, 5, 3)
