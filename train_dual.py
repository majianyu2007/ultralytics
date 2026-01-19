#!/usr/bin/env python3
"""
YOLOv26 双流训练脚本

使用修改后的ultralytics框架训练双流（RGB + IR）YOLO模型

使用方法:
    python train_dual.py --data dual_dataset.yaml --epochs 100 --batch 16
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.models.yolo.detect.dual_train import DualStreamDetectionTrainer
from ultralytics.utils import LOGGER


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv26双流模型训练')

    # 基本参数
    parser.add_argument('--model', type=str, default='ultralytics/cfg/models/26/yolo26.yaml',
                       help='模型配置文件路径')
    parser.add_argument('--data', type=str, required=True,
                       help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='图像尺寸')

    # 设备参数
    parser.add_argument('--device', type=str, default='',
                       help='训练设备 (例: 0, 1,2,3 或 cpu)')

    # 训练参数
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='初始学习率')
    parser.add_argument('--momentum', type=float, default=0.937,
                       help='动量')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='权重衰减')
    parser.add_argument('--optimizer', type=str, default='auto',
                       help='优化器 (auto, SGD, Adam, AdamW, RMSProp)')

    # 数据增强
    parser.add_argument('--hsv-h', type=float, default=0.015,
                       help='HSV色调增强')
    parser.add_argument('--hsv-s', type=float, default=0.7,
                       help='HSV饱和度增强')
    parser.add_argument('--hsv-v', type=float, default=0.4,
                       help='HSV明度增强')

    # 输出设置
    parser.add_argument('--project', type=str, default='runs/train',
                       help='项目保存目录')
    parser.add_argument('--name', type=str, default='dual_stream_exp',
                       help='实验名称')
    parser.add_argument('--save-period', type=int, default=-1,
                       help='保存检查点的间隔 (-1表示仅保存最终模型)')

    # 验证设置
    parser.add_argument('--val', action='store_true',
                       help='训练期间进行验证')
    parser.add_argument('--save-txt', action='store_true',
                       help='保存推理结果为文本文件')
    parser.add_argument('--save-conf', action='store_true',
                       help='保存置信度到文本文件')

    # 其他设置
    parser.add_argument('--workers', type=int, default=8,
                       help='数据加载器的工作进程数')
    parser.add_argument('--seed', type=int, default=0,
                       help='全局随机种子')
    parser.add_argument('--deterministic', action='store_true',
                       help='强制确定性算法')
    parser.add_argument('--single-cls', action='store_true',
                       help='训练为单类别模型')
    parser.add_argument('--rect', action='store_true',
                       help='矩形训练')
    parser.add_argument('--cos-lr', action='store_true',
                       help='余弦学习率调度')
    parser.add_argument('--close-mosaic', type=int, default=10,
                       help='关闭马赛克增强的轮数')

    # 预训练权重
    parser.add_argument('--pretrained', type=str, default='',
                       help='预训练权重路径')
    parser.add_argument('--freeze', nargs='*', type=int, default=None,
                       help='冻结层数 (backbone=10, first3=0 1 2)')

    return parser.parse_args()


def main():
    """主训练函数"""
    args = parse_args()

    LOGGER.info("开始YOLOv26双流模型训练")
    LOGGER.info("=" * 80)

    # 验证数据配置文件
    if not Path(args.data).exists():
        LOGGER.error(f"数据配置文件不存在: {args.data}")
        sys.exit(1)

    # 验证模型配置文件
    if not Path(args.model).exists():
        LOGGER.error(f"模型配置文件不存在: {args.model}")
        sys.exit(1)

    # 构建训练配置
    train_args = {
        'model': args.model,
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'lr0': args.lr0,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'project': args.project,
        'name': args.name,
        'save_period': args.save_period,
        'val': args.val,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'workers': args.workers,
        'seed': args.seed,
        'deterministic': args.deterministic,
        'single_cls': args.single_cls,
        'rect': args.rect,
        'cos_lr': args.cos_lr,
        'close_mosaic': args.close_mosaic,
    }

    # 添加预训练权重
    if args.pretrained:
        train_args['pretrained'] = args.pretrained

    # 添加冻结层设置
    if args.freeze is not None:
        train_args['freeze'] = args.freeze

    # 显示配置信息
    LOGGER.info("训练配置:")
    for key, value in train_args.items():
        LOGGER.info(f"  {key}: {value}")
    LOGGER.info("=" * 80)

    try:
        # 创建双流训练器
        trainer = DualStreamDetectionTrainer(overrides=train_args)

        # 开始训练
        results = trainer.train()

        # 训练完成
        LOGGER.info("=" * 80)
        LOGGER.info("训练完成！")
        LOGGER.info(f"最佳权重保存在: {trainer.best}")
        LOGGER.info(f"最终权重保存在: {trainer.last}")

        # 显示训练结果
        if hasattr(results, 'results_dict'):
            LOGGER.info("训练结果:")
            for metric, value in results.results_dict.items():
                LOGGER.info(f"  {metric}: {value}")

    except Exception as e:
        LOGGER.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
