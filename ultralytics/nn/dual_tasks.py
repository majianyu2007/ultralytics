# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Dual-stream YOLO model wrapper for 6-channel (RGB + IR) inputs."""

from __future__ import annotations

import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import feature_visualization


class DualStreamDetectionModel(DetectionModel):
    """Detection model that enforces 6-channel inputs (RGB + IR stacked)."""

    def __init__(self, cfg="ultralytics/cfg/models/26/yolo26.yaml", ch=6, nc=None, verbose=True):
        if ch != 6:
            LOGGER.warning(f"Dual-stream model expects 6 input channels, got ch={ch}. Forcing ch=6.")
            ch = 6
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.is_dual_stream = True


class DualStreamTransformerDetectionModel(DetectionModel):
    """Dual-stream detection model with transformer fusion between RGB and IR streams."""

    def __init__(self, cfg="ultralytics/cfg/models/26/yolo26-dual-transformer.yaml", ch=3, nc=None, verbose=True):
        if ch != 3:
            LOGGER.warning(f"Dual-stream transformer model expects 3-channel inputs per stream, got ch={ch}.")
            ch = 3
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.is_dual_stream = True
        self.dual_fusion = "transformer"

    @staticmethod
    def _split_inputs(x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor input, got shape {tuple(x.shape)}")
        if x.shape[1] == 6:
            return x[:, :3, :, :], x[:, 3:, :, :]
        if x.shape[1] == 3:
            return x, torch.zeros_like(x)
        raise ValueError(f"Expected 3 or 6 input channels, got {x.shape[1]}")

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        rgb, ir = self._split_inputs(x)
        return self._predict_once_dual(rgb, ir, profile=profile, visualize=visualize, embed=embed)

    def _predict_once_dual(self, x, x2, profile=False, visualize=False, embed=None):
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:
                if isinstance(m.f, int):
                    if m.f == -4:
                        x = x2
                    else:
                        x = y[m.f]
                else:
                    x = [x if j == -1 else (x2 if j == -4 else y[j]) for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize and isinstance(x, torch.Tensor):
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed and isinstance(x, torch.Tensor):
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x
