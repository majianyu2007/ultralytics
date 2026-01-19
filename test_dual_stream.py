#!/usr/bin/env python3
"""
YOLOv26 双流模型测试脚本

这个脚本演示了如何使用修改后的ultralytics进行双流（RGB + IR）目标检测的训练和推理。

使用方法：
1. 准备数据集（RGB和IR图像对）
2. 修改数据配置文件路径
3. 运行此脚本

作者: Claude Code
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ultralytics.models.yolo.detect.dual_train import DualStreamDetectionTrainer
from ultralytics.nn.dual_tasks import DualStreamDetectionModel
from ultralytics.data.dual_dataset import DualStreamYOLODataset
from ultralytics.utils import LOGGER


def create_dummy_data(output_dir="dummy_dual_data", num_images=10):
    """
    创建虚拟双流数据用于测试

    Args:
        output_dir (str): 输出目录
        num_images (int): 生成的图像对数量

    Returns:
        str: 数据集配置文件路径
    """
    import cv2
    import yaml

    output_path = Path(output_dir)

    # 创建目录结构
    rgb_train_dir = output_path / "rgb" / "train"
    ir_train_dir = output_path / "ir" / "train"
    rgb_val_dir = output_path / "rgb" / "val"
    ir_val_dir = output_path / "ir" / "val"
    labels_train_dir = output_path / "labels" / "train"
    labels_val_dir = output_path / "labels" / "val"

    for dir_path in [rgb_train_dir, ir_train_dir, rgb_val_dir, ir_val_dir, labels_train_dir, labels_val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"创建虚拟数据集在: {output_path}")

    # 生成训练数据
    for i in range(num_images):
        # 创建虚拟RGB图像 (640x640, 3通道)
        rgb_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(rgb_train_dir / f"image_{i:03d}.jpg"), rgb_img)
        cv2.imwrite(str(rgb_val_dir / f"image_{i:03d}.jpg"), rgb_img)

        # 创建虚拟IR图像 (640x640, 3通道, 但内容不同)
        ir_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(ir_train_dir / f"image_{i:03d}.jpg"), ir_img)
        cv2.imwrite(str(ir_val_dir / f"image_{i:03d}.jpg"), ir_img)

        # 创建虚拟标签文件
        label_content = "0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.15 0.25\n"  # class x y w h (normalized)
        with open(labels_train_dir / f"image_{i:03d}.txt", 'w') as f:
            f.write(label_content)
        with open(labels_val_dir / f"image_{i:03d}.txt", 'w') as f:
            f.write(label_content)

    # 创建数据集配置文件
    dataset_config = {
        'rgb_train': str(rgb_train_dir),
        'rgb_val': str(rgb_val_dir),
        'ir_train': str(ir_train_dir),
        'ir_val': str(ir_val_dir),
        'train': str(rgb_train_dir),
        'val': str(rgb_val_dir),
        'labels_train': str(labels_train_dir),
        'labels_val': str(labels_val_dir),
        'nc': 2,  # 2个类别用于测试
        'names': ['person', 'vehicle']
    }

    config_path = output_path / "dataset_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    LOGGER.info(f"数据集配置保存到: {config_path}")
    return str(config_path)


def test_dual_stream_dataset():
    """测试双流数据集加载"""
    LOGGER.info("=" * 60)
    LOGGER.info("测试1: 双流数据集加载")
    LOGGER.info("=" * 60)

    try:
        # 创建虚拟数据
        config_path = create_dummy_data()

        # 测试数据集加载
        dataset = DualStreamYOLODataset(
            rgb_img_path="dummy_dual_data/rgb/train",
            ir_img_path="dummy_dual_data/ir/train",
            imgsz=640,
            augment=False,
            label_path="dummy_dual_data/labels/train",
        )

        LOGGER.info(f"数据集大小: {len(dataset)}")

        # 测试获取一个样本
        sample = dataset[0]
        img_tensor = sample['img']

        LOGGER.info(f"输入图像shape: {img_tensor.shape}")  # 应该是 (6, 640, 640)
        LOGGER.info(f"前3通道 (RGB) 范围: [{img_tensor[:3].min():.3f}, {img_tensor[:3].max():.3f}]")
        LOGGER.info(f"后3通道 (IR) 范围: [{img_tensor[3:].min():.3f}, {img_tensor[3:].max():.3f}]")

        if img_tensor.shape[0] == 6:
            LOGGER.info("✅ 双流数据集加载成功！")
            return True
        else:
            LOGGER.error(f"❌ 错误：期望6通道，但得到{img_tensor.shape[0]}通道")
            return False

    except Exception as e:
        LOGGER.error(f"❌ 双流数据集测试失败: {e}")
        return False


def test_dual_stream_model():
    """测试双流模型"""
    LOGGER.info("=" * 60)
    LOGGER.info("测试2: 双流模型推理")
    LOGGER.info("=" * 60)

    try:
        # 创建双流模型
        model = DualStreamDetectionModel(cfg="ultralytics/cfg/models/26/yolo26.yaml", ch=6, nc=2)
        model.eval()

        # 创建虚拟输入 (batch_size=2, channels=6, height=640, width=640)
        dummy_input = torch.randn(2, 6, 640, 640)

        LOGGER.info(f"输入shape: {dummy_input.shape}")

        # 前向传播
        with torch.no_grad():
            output = model(dummy_input)

        LOGGER.info(f"输出类型: {type(output)}")
        if isinstance(output, torch.Tensor):
            LOGGER.info(f"输出shape: {output.shape}")
        elif isinstance(output, (list, tuple)):
            LOGGER.info(f"输出数量: {len(output)}")
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    LOGGER.info(f"输出 {i} shape: {out.shape}")

        LOGGER.info("✅ 双流模型推理成功！")
        return True

    except Exception as e:
        LOGGER.error(f"❌ 双流模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dual_stream_training():
    """测试双流模型训练"""
    LOGGER.info("=" * 60)
    LOGGER.info("测试3: 双流模型训练")
    LOGGER.info("=" * 60)

    try:
        # 创建虚拟数据
        config_path = create_dummy_data()

        # 配置训练参数
        args = {
            'model': 'ultralytics/cfg/models/26/yolo26.yaml',
            'data': config_path,
            'epochs': 2,  # 只训练2个epoch用于测试
            'batch': 2,   # 小批次
            'imgsz': 640,
            'save': True,
            'verbose': True,
            'device': 'cpu',  # 使用CPU避免GPU内存问题
        }

        # 创建训练器
        trainer = DualStreamDetectionTrainer(overrides=args)

        LOGGER.info("开始训练...")

        # 运行训练（只训练几步用于测试）
        trainer.train()

        LOGGER.info("✅ 双流模型训练成功！")
        return True

    except Exception as e:
        LOGGER.error(f"❌ 双流训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_splitting():
    """测试6通道数据的分离"""
    LOGGER.info("=" * 60)
    LOGGER.info("测试4: 6通道数据分离")
    LOGGER.info("=" * 60)

    try:
        # 创建6通道测试数据
        batch_size, height, width = 4, 640, 640
        test_data = torch.randn(batch_size, 6, height, width)

        # 分离RGB和IR
        rgb_data = test_data[:, :3, :, :]
        ir_data = test_data[:, 3:, :, :]

        LOGGER.info(f"原始数据 shape: {test_data.shape}")
        LOGGER.info(f"RGB数据 shape: {rgb_data.shape}")
        LOGGER.info(f"IR数据 shape: {ir_data.shape}")

        # 验证分离是否正确
        assert rgb_data.shape == (batch_size, 3, height, width), f"RGB shape错误: {rgb_data.shape}"
        assert ir_data.shape == (batch_size, 3, height, width), f"IR shape错误: {ir_data.shape}"

        # 验证数据一致性
        reconstructed = torch.cat([rgb_data, ir_data], dim=1)
        assert torch.allclose(test_data, reconstructed), "数据重构不一致"

        LOGGER.info("✅ 6通道数据分离测试成功！")
        return True

    except Exception as e:
        LOGGER.error(f"❌ 数据分离测试失败: {e}")
        return False


def main():
    """主测试函数"""
    LOGGER.info("开始YOLOv26双流模型测试")
    LOGGER.info("=" * 80)

    test_results = []

    # 运行所有测试
    test_results.append(("数据分离测试", test_data_splitting()))
    test_results.append(("双流数据集测试", test_dual_stream_dataset()))
    test_results.append(("双流模型测试", test_dual_stream_model()))
    test_results.append(("双流训练测试", test_dual_stream_training()))

    # 汇总结果
    LOGGER.info("=" * 80)
    LOGGER.info("测试结果汇总:")
    LOGGER.info("=" * 80)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        LOGGER.info(f"{test_name}: {status}")
        if result:
            passed += 1

    LOGGER.info("=" * 80)
    LOGGER.info(f"总计: {passed}/{total} 测试通过")

    if passed == total:
        LOGGER.info("🎉 所有测试通过！YOLOv26双流改造成功！")

        # 输出使用说明
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("使用说明:")
        LOGGER.info("=" * 80)
        LOGGER.info("1. 准备你的双流数据集（RGB + IR图像对）")
        LOGGER.info("2. 创建数据配置文件，参考 dual_dataset_example.yaml")
        LOGGER.info("3. 使用以下命令训练:")
        LOGGER.info("   python -c \"from dual_train import DualStreamDetectionTrainer; trainer = DualStreamDetectionTrainer(overrides={'model': 'ultralytics/cfg/models/26/yolo26.yaml', 'data': 'your_data.yaml', 'epochs': 100}); trainer.train()\"")
        LOGGER.info("4. 训练完成后，模型将自动保存")

    else:
        LOGGER.error(f"❌ {total - passed} 个测试失败，请检查错误信息")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
