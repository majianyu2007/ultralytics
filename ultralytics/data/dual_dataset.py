# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import HELP_URL, get_hash, img2label_paths, load_dataset_cache_file
from ultralytics.utils import LOCAL_RANK, LOGGER, TQDM
from ultralytics.utils.patches import imread


class DualStreamYOLODataset(YOLODataset):
    """Dataset for dual-stream (RGB + IR) inputs that are stacked into 6 channels."""

    def __init__(
        self,
        rgb_img_path: str,
        ir_img_path: str,
        *args: Any,
        data: dict | None = None,
        label_path: str | None = None,
        **kwargs: Any,
    ):
        self.rgb_img_path = rgb_img_path
        self.ir_img_path = ir_img_path
        self.label_path = label_path
        data = {} if data is None else data
        data["channels"] = 6
        super().__init__(img_path=rgb_img_path, data=data, *args, **kwargs)
        self.ir_files = self._match_ir_files()

    def _build_label_files(self) -> list[str]:
        if not self.label_path:
            return img2label_paths(self.im_files)

        label_root = Path(self.label_path)
        rgb_root = Path(self.rgb_img_path)
        label_files = []
        for rgb_file in self.im_files:
            rgb_path = Path(rgb_file)
            if rgb_root in rgb_path.parents:
                rel_path = rgb_path.relative_to(rgb_root)
                label_file = label_root / rel_path
            else:
                label_file = label_root / rgb_path.name
            label_files.append(label_file.with_suffix(".txt").as_posix())
        return label_files

    def get_labels(self) -> list[dict]:
        self.label_files = self._build_label_files()
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, exists = self.cache_labels(cache_path), False

        nf, nm, ne, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))

        [cache.pop(k) for k in ("hash", "version", "msgs")]
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]

        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def _match_ir_files(self) -> list[str]:
        """Match RGB image files to IR image files by relative path or filename."""
        rgb_root = Path(self.rgb_img_path)
        ir_root = Path(self.ir_img_path)
        ir_files = []
        missing = []

        for rgb_file in self.im_files:
            rgb_path = Path(rgb_file)
            candidate = None
            if rgb_root in rgb_path.parents:
                rel_path = rgb_path.relative_to(rgb_root)
                rel_candidate = ir_root / rel_path
                if rel_candidate.exists():
                    candidate = rel_candidate
            if candidate is None:
                name_candidate = ir_root / rgb_path.name
                if name_candidate.exists():
                    candidate = name_candidate

            if candidate is None:
                missing.append(str(rgb_path))
            else:
                ir_files.append(str(candidate))

        if missing:
            missing_preview = "\n".join(missing[:10])
            raise FileNotFoundError(
                "Missing IR images for the following RGB files (showing up to 10):\n"
                f"{missing_preview}"
            )

        return ir_files

    def _load_stream_image(self, path: Path) -> np.ndarray:
        """Load a single stream image in BGR format."""
        img = imread(str(path), flags=cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image Not Found {path}")
        if img.ndim == 2:
            img = img[..., None]
        return img

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load a dual-stream image from dataset index 'i'."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:
            if fn.exists():
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = None

            if im is None:
                rgb = self._load_stream_image(Path(f))
                ir = self._load_stream_image(Path(self.ir_files[i]))

                if rgb.shape[:2] != ir.shape[:2]:
                    LOGGER.warning(
                        f"{self.prefix}IR image shape {ir.shape[:2]} does not match RGB {rgb.shape[:2]} for {f}. "
                        "Resizing IR to match RGB."
                    )
                    ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

                h0, w0 = rgb.shape[:2]
                im = np.concatenate([rgb, ir], axis=2)
            else:
                h0, w0 = im.shape[:2]

            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (min(int(w0 * r + 0.5), self.imgsz), min(int(h0 * r + 0.5), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images_to_disk(self, i: int) -> None:
        """Save a dual-stream image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if f.exists():
            return
        rgb = self._load_stream_image(Path(self.im_files[i]))
        ir = self._load_stream_image(Path(self.ir_files[i]))
        if rgb.shape[:2] != ir.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        np.save(f.as_posix(), np.concatenate([rgb, ir], axis=2), allow_pickle=False)
