#!/usr/bin/env python3
"""
YOLOv26 双流推理脚本

使用训练好的双流YOLO模型进行RGB + IR图像的目标检测推理

使用方法:
    python detect_dual.py --weights best.pt --rgb-source rgb_images/ --ir-source ir_images/ --save-dir results/
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.nn.dual_tasks import DualStreamDetectionModel, DualStreamTransformerDetectionModel
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator, colors


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv26双流模型推理')

    # 输入参数
    parser.add_argument('--weights', type=str, required=True,
                       help='模型权重文件路径')
    parser.add_argument('--model', type=str, default='',
                       help='模型配置文件路径（用于双流Transformer融合）')
    parser.add_argument('--rgb-source', type=str, required=True,
                       help='RGB图像路径（文件夹或单个图像）')
    parser.add_argument('--ir-source', type=str, required=True,
                       help='IR图像路径（文件夹或单个图像）')

    # 推理参数
    parser.add_argument('--imgsz', type=int, default=640,
                       help='推理图像尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                       help='NMS IoU阈值')
    parser.add_argument('--max-det', type=int, default=1000,
                       help='每张图像最大检测数')

    # 输出参数
    parser.add_argument('--save-dir', type=str, default='runs/detect/dual_stream_exp',
                       help='结果保存目录')
    parser.add_argument('--save-txt', action='store_true',
                       help='保存结果为文本文件')
    parser.add_argument('--save-conf', action='store_true',
                       help='在文本文件中保存置信度')
    parser.add_argument('--save-crop', action='store_true',
                       help='保存裁剪的检测框')
    parser.add_argument('--nosave', action='store_true',
                       help='不保存图像')
    parser.add_argument('--view-img', action='store_true',
                       help='显示推理结果')

    # 设备参数
    parser.add_argument('--device', type=str, default='',
                       help='推理设备 (例: 0, cpu)')

    # 其他参数
    parser.add_argument('--classes', nargs='+', type=int,
                       help='按类别过滤: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                       help='类别无关NMS')
    parser.add_argument('--augment', action='store_true',
                       help='增强推理')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化特征')
    parser.add_argument('--line-thickness', type=int, default=3,
                       help='边界框线条粗细')
    parser.add_argument('--hide-labels', action='store_true',
                       help='隐藏标签')
    parser.add_argument('--hide-conf', action='store_true',
                       help='隐藏置信度')
    parser.add_argument('--half', action='store_true',
                       help='使用FP16半精度推理')

    return parser.parse_args()


def load_paired_images(rgb_source, ir_source):
    """
    加载配对的RGB和IR图像

    Args:
        rgb_source (str): RGB图像路径
        ir_source (str): IR图像路径

    Returns:
        list: 配对的图像路径列表 [(rgb_path, ir_path), ...]
    """
    rgb_path = Path(rgb_source)
    ir_path = Path(ir_source)

    paired_images = []

    if rgb_path.is_file() and ir_path.is_file():
        # 单个图像文件
        paired_images.append((str(rgb_path), str(ir_path)))
    elif rgb_path.is_dir() and ir_path.is_dir():
        # 图像文件夹
        rgb_images = sorted(list(rgb_path.glob('*.jpg')) + list(rgb_path.glob('*.png')) + list(rgb_path.glob('*.jpeg')))
        ir_images = sorted(list(ir_path.glob('*.jpg')) + list(ir_path.glob('*.png')) + list(ir_path.glob('*.jpeg')))

        # 按文件名配对
        rgb_dict = {img.stem: img for img in rgb_images}
        ir_dict = {img.stem: img for img in ir_images}

        common_names = set(rgb_dict.keys()) & set(ir_dict.keys())

        if not common_names:
            raise ValueError("RGB和IR文件夹中没有找到匹配的图像对")

        for name in sorted(common_names):
            paired_images.append((str(rgb_dict[name]), str(ir_dict[name])))

    else:
        raise ValueError("RGB和IR源必须都是文件或都是文件夹")

    return paired_images


def preprocess_dual_image(rgb_img, ir_img, imgsz=640):
    """
    预处理双流图像

    Args:
        rgb_img (np.ndarray): RGB图像
        ir_img (np.ndarray): IR图像
        imgsz (int): 目标尺寸

    Returns:
        torch.Tensor: 预处理后的6通道张量
    """
    # 对齐大小
    if rgb_img.shape[:2] != ir_img.shape[:2]:
        ir_img = cv2.resize(ir_img, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 调整大小
    h, w = rgb_img.shape[:2]
    r = imgsz / max(h, w)
    if r != 1:
        new_size = (int(w * r), int(h * r))
        rgb_img = cv2.resize(rgb_img, new_size)
        ir_img = cv2.resize(ir_img, new_size)

    # 填充为正方形
    h, w = rgb_img.shape[:2]
    pad_h = (imgsz - h) // 2
    pad_w = (imgsz - w) // 2

    rgb_img = cv2.copyMakeBorder(rgb_img, pad_h, imgsz - h - pad_h, pad_w, imgsz - w - pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    ir_img = cv2.copyMakeBorder(ir_img, pad_h, imgsz - h - pad_h, pad_w, imgsz - w - pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # 归一化和转换为张量
    rgb_img = rgb_img.transpose(2, 0, 1) / 255.0  # HWC to CHW, 0-255 to 0-1
    ir_img = ir_img.transpose(2, 0, 1) / 255.0

    # 拼接为6通道
    dual_img = np.concatenate([rgb_img, ir_img], axis=0)  # (6, H, W)

    return torch.from_numpy(dual_img).float(), (r, pad_h, pad_w)


def postprocess_predictions(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000):
    """后处理预测结果"""
    from ultralytics.utils.ops import non_max_suppression

    # 应用NMS
    pred = non_max_suppression(
        pred,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=classes,
        agnostic=agnostic,
        max_det=max_det
    )

    return pred


def scale_boxes(boxes, img_shape, ratio_pad):
    """将检测框坐标从推理尺寸缩放回原始图像尺寸"""
    r, pad_h, pad_w = ratio_pad

    # 移除填充
    boxes[:, [0, 2]] -= pad_w  # x padding
    boxes[:, [1, 3]] -= pad_h  # y padding

    # 缩放
    boxes /= r

    # 裁剪边界
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img_shape[1])  # x
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img_shape[0])  # y

    return boxes


def main():
    """主推理函数"""
    args = parse_args()

    LOGGER.info("开始YOLOv26双流模型推理")
    LOGGER.info("=" * 80)

    # 检查权重文件
    if not Path(args.weights).exists():
        LOGGER.error(f"权重文件不存在: {args.weights}")
        sys.exit(1)

    # 设置设备
    device = select_device(args.device)
    LOGGER.info(f"使用设备: {device}")

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 加载模型
        LOGGER.info(f"加载模型: {args.weights}")
        weights = torch.load(args.weights, map_location=device)

        cfg_for_model = args.model if args.model else None
        if not cfg_for_model and isinstance(weights, dict):
            model_obj = weights.get("model")
            if hasattr(model_obj, "yaml"):
                cfg_for_model = model_obj.yaml

        use_transformer = False
        if isinstance(cfg_for_model, dict):
            use_transformer = cfg_for_model.get("dual_fusion") == "transformer"
        elif cfg_for_model:
            try:
                cfg_for_model = yaml_model_load(cfg_for_model)
                use_transformer = cfg_for_model.get("dual_fusion") == "transformer"
            except Exception:
                pass

        if use_transformer:
            model = DualStreamTransformerDetectionModel(cfg_for_model, ch=3)
        else:
            model = DualStreamDetectionModel(cfg_for_model or "ultralytics/cfg/models/26/yolo26.yaml", ch=6)

        model.load(weights)
        model = model.to(device)
        model.eval()

        if args.half and device.type != 'cpu':
            model.half()

        # 获取类别名称
        names = model.names if hasattr(model, 'names') else {i: f'class{i}' for i in range(1000)}

        # 加载图像对
        LOGGER.info("加载图像对...")
        paired_images = load_paired_images(args.rgb_source, args.ir_source)
        LOGGER.info(f"找到 {len(paired_images)} 对图像")

        # 推理
        total_time = 0
        for i, (rgb_path, ir_path) in enumerate(paired_images):
            LOGGER.info(f"处理 {i+1}/{len(paired_images)}: {Path(rgb_path).name}, {Path(ir_path).name}")

            # 加载图像
            rgb_img = cv2.imread(rgb_path)
            ir_img = cv2.imread(ir_path)

            if rgb_img is None or ir_img is None:
                LOGGER.error(f"无法加载图像: {rgb_path} 或 {ir_path}")
                continue

            orig_shape = rgb_img.shape[:2]

            # 预处理
            dual_tensor, ratio_pad = preprocess_dual_image(rgb_img, ir_img, args.imgsz)
            dual_tensor = dual_tensor.unsqueeze(0).to(device)  # 添加批次维度

            if args.half and device.type != 'cpu':
                dual_tensor = dual_tensor.half()

            # 推理
            t1 = time.time()
            with torch.no_grad():
                pred = model(dual_tensor, augment=args.augment, visualize=args.visualize)
            t2 = time.time()

            inference_time = t2 - t1
            total_time += inference_time

            # 后处理
            pred = postprocess_predictions(
                pred,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                classes=args.classes,
                agnostic=args.agnostic_nms,
                max_det=args.max_det
            )

            # 处理检测结果
            det = pred[0]  # 第一个（也是唯一的）图像的检测结果

            # 打印结果
            LOGGER.info(f"推理时间: {inference_time:.3f}s, 检测到 {len(det)} 个目标")

            if len(det):
                # 缩放边界框到原始图像尺寸
                det[:, :4] = scale_boxes(det[:, :4], orig_shape, ratio_pad)

                # 绘制结果（在RGB图像上）
                annotator = Annotator(rgb_img, line_width=args.line_thickness, example=str(names))

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = None if args.hide_labels else (names[c] if args.hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

                # 保存结果
                if not args.nosave:
                    result_path = save_dir / f'result_{Path(rgb_path).stem}.jpg'
                    cv2.imwrite(str(result_path), rgb_img)

                # 保存文本结果
                if args.save_txt:
                    txt_path = save_dir / f'result_{Path(rgb_path).stem}.txt'
                    with open(txt_path, 'w') as f:
                        for *xyxy, conf, cls in det:
                            if args.save_conf:
                                f.write(f"{int(cls)} {' '.join(map(str, xyxy))} {conf}\n")
                            else:
                                f.write(f"{int(cls)} {' '.join(map(str, xyxy))}\n")

                # 显示结果
                if args.view_img:
                    cv2.imshow('Results', rgb_img)
                    if cv2.waitKey(1) == ord('q'):
                        break

        # 统计信息
        LOGGER.info("=" * 80)
        LOGGER.info(f"推理完成！")
        LOGGER.info(f"处理了 {len(paired_images)} 对图像")
        LOGGER.info(f"平均推理时间: {total_time/len(paired_images):.3f}s")
        LOGGER.info(f"结果保存在: {save_dir}")

        if args.view_img:
            cv2.destroyAllWindows()

    except Exception as e:
        LOGGER.error(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
