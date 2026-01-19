#!/usr/bin/env python3
"""Prepare M3FD dual-stream subset in YOLO format (RGB + IR)."""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


CLASSES = ["Bus", "Car", "Lamp", "Motorcycle", "People", "Truck"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare M3FD subset for dual-stream YOLO training.")
    parser.add_argument("--src-root", type=Path, required=True, help="M3FD root with Vis/Ir/Annotation")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root for split dataset")
    parser.add_argument("--yaml-out", type=Path, required=True, help="Output YAML path")
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--val-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dirs(root: Path) -> None:
    for sub in [
        root / "rgb" / "train",
        root / "rgb" / "val",
        root / "ir" / "train",
        root / "ir" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]:
        sub.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def convert_one(stem: str, split: str, ann_dir: Path, vis_dir: Path, ir_dir: Path, out_root: Path, cls_map: dict[str, int]) -> None:
    xml_path = ann_dir / f"{stem}.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing size in {xml_path}")
    w = int(size.findtext("width"))
    h = int(size.findtext("height"))

    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in cls_map:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))
        x = (xmin + xmax) / 2.0 / w
        y = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{cls_map[name]} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

    label_path = out_root / "labels" / split / f"{stem}.txt"
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    vis_src = vis_dir / f"{stem}.png"
    ir_src = ir_dir / f"{stem}.png"
    vis_dst = out_root / "rgb" / split / f"{stem}.png"
    ir_dst = out_root / "ir" / split / f"{stem}.png"
    link_or_copy(vis_src, vis_dst)
    link_or_copy(ir_src, ir_dst)


def write_yaml(path: Path, data_root: Path) -> None:
    content = (
        "# M3FD dual-stream dataset (subset for quick validation)\n"
        f"path: {data_root}\n\n"
        "rgb_train: rgb/train\n"
        "rgb_val: rgb/val\n"
        "ir_train: ir/train\n"
        "ir_val: ir/val\n\n"
        "train: rgb/train\n"
        "val: rgb/val\n\n"
        "labels_train: labels/train\n"
        "labels_val: labels/val\n\n"
        "nc: 6\n"
        "names: ['Bus', 'Car', 'Lamp', 'Motorcycle', 'People', 'Truck']\n"
    )
    path.write_text(content)


def main() -> None:
    args = parse_args()

    vis_dir = args.src_root / "Vis"
    ir_dir = args.src_root / "Ir"
    ann_dir = args.src_root / "Annotation"

    if not (vis_dir.exists() and ir_dir.exists() and ann_dir.exists()):
        raise FileNotFoundError("M3FD source must contain Vis/Ir/Annotation")

    ensure_dirs(args.out_root)

    stems = sorted(p.stem for p in ann_dir.glob("*.xml"))
    if not stems:
        raise FileNotFoundError(f"No annotations found in {ann_dir}")

    random.seed(args.seed)
    random.shuffle(stems)

    total_needed = args.train_count + args.val_count
    if total_needed > len(stems):
        raise ValueError(f"Requested {total_needed} samples, but only {len(stems)} available")

    train_stems = stems[: args.train_count]
    val_stems = stems[args.train_count : args.train_count + args.val_count]

    cls_map = {name: i for i, name in enumerate(CLASSES)}

    for stem in train_stems:
        convert_one(stem, "train", ann_dir, vis_dir, ir_dir, args.out_root, cls_map)
    for stem in val_stems:
        convert_one(stem, "val", ann_dir, vis_dir, ir_dir, args.out_root, cls_map)

    write_yaml(args.yaml_out, args.out_root)
    print(f"Prepared M3FD subset: train={len(train_stems)} val={len(val_stems)}")
    print(f"YAML: {args.yaml_out}")


if __name__ == "__main__":
    main()
