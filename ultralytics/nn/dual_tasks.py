# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Dual-stream YOLO model wrapper for 6-channel (RGB + IR) inputs."""

from __future__ import annotations

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER


class DualStreamDetectionModel(DetectionModel):
    """Detection model that enforces 6-channel inputs (RGB + IR stacked)."""

    def __init__(self, cfg="ultralytics/cfg/models/26/yolo26.yaml", ch=6, nc=None, verbose=True):
        if ch != 6:
            LOGGER.warning(f"Dual-stream model expects 6 input channels, got ch={ch}. Forcing ch=6.")
            ch = 6
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.is_dual_stream = True
