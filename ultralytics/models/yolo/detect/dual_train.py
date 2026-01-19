# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
双流YOLO检测训练器

扩展标准的检测训练器以支持双流（RGB + IR）输入的训练
"""

from __future__ import annotations

import math
import random
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader
from ultralytics.data.build import build_dual_stream_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.dual_tasks import DualStreamDetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class DualStreamDetectionTrainer(BaseTrainer):
    """
    双流YOLO检测训练器

    该训练器专门用于训练双流YOLO模型，处理RGB和IR图像的双模态输入。
    支持标准YOLO训练的所有功能，同时添加了双流特定的数据处理和模型管理。

    主要特点：
    - 支持RGB+IR双流数据加载
    - 自动处理6通道输入的预处理
    - 兼容标准YOLO训练流程
    - 支持双流特定的可视化

    Attributes:
        model (DualStreamDetectionModel): 双流检测模型
        data (dict): 包含RGB和IR数据路径的数据集配置
        loss_names (tuple): 损失组件名称

    Examples:
        >>> from ultralytics.models.yolo.detect.dual_train import DualStreamDetectionTrainer
        >>> args = dict(model="yolo26n.pt", data="dual_dataset.yaml", epochs=100)
        >>> trainer = DualStreamDetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """
        初始化双流检测训练器

        Args:
            cfg (dict): 默认配置字典
            overrides (dict): 参数覆盖字典
            _callbacks (list): 回调函数列表
        """
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """
        构建双流YOLO数据集

        Args:
            img_path (str): 图像文件夹路径（对于双流，这个参数会被data配置中的路径覆盖）
            mode (str): 训练模式 ('train' 或 'val')
            batch (int): 批次大小

        Returns:
            Dataset: 配置好的双流数据集
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        data_root = Path(self.data.get("path")) if self.data and self.data.get("path") else None

        def _resolve_path(value):
            if value is None:
                return value
            if isinstance(value, (list, tuple)):
                return [_resolve_path(v) for v in value]
            p = Path(value)
            if p.is_absolute() or data_root is None:
                return str(p)
            candidate = (data_root / p).resolve()
            if not candidate.exists() and str(value).startswith("../"):
                candidate = (data_root / str(value)[3:]).resolve()
            return str(candidate)

        # 检查是否有双流数据路径配置
        if 'rgb_' + mode in self.data and 'ir_' + mode in self.data:
            # 双流模式：分别指定RGB和IR路径
            rgb_key = f"rgb_{mode}"
            ir_key = f"ir_{mode}"
            labels_key = f"labels_{mode}"
            rgb_path = _resolve_path(self.data[rgb_key])
            ir_path = _resolve_path(self.data[ir_key])
            self.data[rgb_key] = rgb_path
            self.data[ir_key] = ir_path
            if self.data.get(labels_key):
                self.data[labels_key] = _resolve_path(self.data[labels_key])

            LOGGER.info(f"构建双流数据集 - RGB: {rgb_path}, IR: {ir_path}")

            return build_dual_stream_dataset(
                self.args,
                rgb_path,
                ir_path,
                batch,
                self.data,
                mode=mode,
                rect=mode == "val",
                stride=gs
            )
        else:
            # 如果没有双流配置，尝试从标准路径推断
            LOGGER.warning("未找到双流数据配置 (rgb_train, ir_train 等)，尝试从标准路径推断...")

            # 假设标准路径下有rgb和ir子文件夹
            base_path = Path(img_path)
            rgb_path = base_path / 'rgb'
            ir_path = base_path / 'ir'

            if rgb_path.exists() and ir_path.exists():
                LOGGER.info(f"自动检测到双流数据 - RGB: {rgb_path}, IR: {ir_path}")
                return build_dual_stream_dataset(
                    self.args,
                    str(rgb_path),
                    str(ir_path),
                    batch,
                    self.data,
                    mode=mode,
                    rect=mode == "val",
                    stride=gs
                )
            else:
                raise ValueError(
                    f"双流数据配置错误！请在数据配置文件中指定:\n"
                    f"rgb_{mode}: /path/to/rgb/images\n"
                    f"ir_{mode}: /path/to/ir/images\n"
                    f"或确保 {img_path} 下存在 'rgb' 和 'ir' 子文件夹"
                )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        构建双流数据加载器

        Args:
            dataset_path (str): 数据集路径
            batch_size (int): 批次大小
            rank (int): 分布式训练进程rank
            mode (str): 模式 ('train' 或 'val')

        Returns:
            DataLoader: PyTorch数据加载器
        """
        assert mode in {"train", "val"}, f"模式必须是 'train' 或 'val'，而不是 {mode}."

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' 与DataLoader shuffle不兼容，设置shuffle=False")
            shuffle = False

        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """
        预处理批次数据，处理6通道输入

        Args:
            batch (dict): 包含图像和标签的批次字典

        Returns:
            dict: 预处理后的批次数据
        """
        # 将数据移动到设备
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")

        # 归一化图像 (0-255 -> 0-1)
        batch["img"] = batch["img"].float() / 255

        # 验证输入维度
        imgs = batch["img"]
        if imgs.shape[1] != 6:
            raise ValueError(f"双流模型期望6通道输入，但收到{imgs.shape[1]}通道")

        # 多尺度训练
        multi_scale = self.args.multi_scale
        if random.random() < multi_scale:
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1 + self.stride))
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs

        return batch

    def set_model_attributes(self):
        """根据数据集信息设置模型属性"""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

        # 确保模型知道它是双流模式
        if hasattr(self.model, 'is_dual_stream'):
            LOGGER.info("模型已配置为双流模式")
        else:
            LOGGER.warning("模型可能不支持双流，但将尝试使用6通道输入")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """
        获取双流检测模型

        Args:
            cfg (str): 模型配置文件路径
            weights (str): 模型权重路径
            verbose (bool): 是否显示详细信息

        Returns:
            DualStreamDetectionModel: 双流检测模型
        """
        # 强制设置为6通道输入
        model = DualStreamDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=6,  # RGB 3 + IR 3
            verbose=verbose and RANK == -1
        )

        if weights:
            model.load(weights)

        # 设置数据集通道数
        self.data["channels"] = 6

        return model

    def get_validator(self):
        """获取双流检测验证器"""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """
        绘制训练样本，分别显示RGB和IR图像

        Args:
            batch (dict): 批次数据
            ni (int): 迭代次数
        """
        imgs = batch["img"]

        # 分离RGB和IR图像用于可视化
        rgb_imgs = imgs[:, :3, :, :] * 255  # RGB前3通道
        ir_imgs = imgs[:, 3:, :, :] * 255   # IR后3通道

        # 创建RGB可视化
        rgb_batch = batch.copy()
        rgb_batch["img"] = rgb_imgs
        plot_images(
            labels=rgb_batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}_rgb.jpg",
            on_plot=self.on_plot,
        )

        # 创建IR可视化（转换为伪彩色显示）
        ir_batch = batch.copy()
        ir_batch["img"] = ir_imgs
        plot_images(
            labels=ir_batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}_ir.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """创建训练标签分布图"""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """
        自动计算最优批次大小

        Returns:
            int: 最优批次大小
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        del train_dataset
        return super().auto_batch(max_num_obj)

    def validate_data_config(self):
        """验证双流数据配置"""
        required_keys = {
            "train": ["rgb_train", "ir_train"],
            "val": ["rgb_val", "ir_val"]
        }

        for split, keys in required_keys.items():
            for key in keys:
                if key not in self.data:
                    raise ValueError(
                        f"双流数据配置缺少 '{key}' 键。\n"
                        f"请确保数据配置文件包含以下键：\n"
                        f"rgb_train: /path/to/rgb/train/images\n"
                        f"ir_train: /path/to/ir/train/images\n"
                        f"rgb_val: /path/to/rgb/val/images\n"
                        f"ir_val: /path/to/ir/val/images"
                    )

    def train(self):
        """运行双流训练"""
        # 验证数据配置
        try:
            self.validate_data_config()
        except ValueError as e:
            LOGGER.warning(f"数据配置验证警告: {e}")

        # 调用父类训练方法
        return super().train()
