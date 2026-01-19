# YOLOv26 双流(RGB + IR)改造说明与使用指南

## 概览

本项目在 Ultralytics YOLOv26 的基础上增加双流输入支持，采用**早期融合**策略：
- RGB(3 通道) + IR(3 通道) 在数据加载阶段拼接为 6 通道
- 模型主干保持标准 YOLOv26 结构，通过 `ch=6` 直接接受 6 通道输入
- 训练、验证、推理流程与原版保持一致

---

## 修改说明（项目级）

### 新增文件
- `ultralytics/data/dual_dataset.py`：双流数据集类（RGB/IR 拼接为 6 通道）
- `ultralytics/nn/dual_tasks.py`：双流检测模型包装器
- `ultralytics/models/yolo/detect/dual_train.py`：双流训练器
- `train_dual.py`：双流训练脚本
- `detect_dual.py`：双流推理脚本
- `dual_dataset_example.yaml`：数据集配置示例
- `test_dual_stream.py`：双流功能测试脚本

### 修改文件
- `ultralytics/data/build.py`：新增 `build_dual_stream_dataset` 数据构建入口

### 关键行为与修复
- **路径解析**：`rgb_* / ir_* / labels_*` 若为相对路径，会自动基于 `data.path` 解析（与 `train/val` 一致）。
- **推理设备参数**：`detect_dual.py` 支持 `--device 0` / `0,1` / `cpu` 等常见写法。
- **标签路径**：可选 `labels_train/labels_val` 指定标签根目录；不提供时按默认 `images -> labels` 规则推断。
- **配对规则**：RGB/IR 通过相对路径或同名文件配对；若尺寸不一致，IR 自动缩放到 RGB 尺寸。

---

## 数据集准备

推荐目录结构如下（RGB/IR 同名配对）：
```
dataset/
├── rgb/
│   ├── train/
│   │   ├── image001.jpg
│   │   └── ...
│   └── val/
│       ├── image001.jpg
│       └── ...
├── ir/
│   ├── train/
│   │   ├── image001.jpg
│   │   └── ...
│   └── val/
│       ├── image001.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image001.txt
    │   └── ...
    └── val/
        ├── image001.txt
        └── ...
```

---

## 数据集配置文件

### 推荐写法（带 `path:` 根目录）
```yaml
# dataset root
path: /path/to/dataset

# RGB/IR 图像路径（相对 path）
rgb_train: rgb/train
rgb_val: rgb/val
ir_train: ir/train
ir_val: ir/val

# YOLO 兼容路径（保持为 RGB）
train: rgb/train
val: rgb/val

# 可选：标签路径（相对 path）
labels_train: labels/train
labels_val: labels/val

nc: 80
names: ['person', 'bicycle', 'car', ...]
```

### 也支持绝对路径
```yaml
rgb_train: /abs/path/to/dataset/rgb/train
rgb_val: /abs/path/to/dataset/rgb/val
ir_train: /abs/path/to/dataset/ir/train
ir_val: /abs/path/to/dataset/ir/val
train: /abs/path/to/dataset/rgb/train
val: /abs/path/to/dataset/rgb/val
labels_train: /abs/path/to/dataset/labels/train
labels_val: /abs/path/to/dataset/labels/val
```

> 提示：若未设置 `labels_*`，将默认从 RGB 图像路径推断标签路径（`images -> labels`）。

---

## 训练使用

```bash
python train_dual.py \
  --model ultralytics/cfg/models/26/yolo26.yaml \
  --data your_dual_dataset.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

可用设备示例：`--device 0`、`--device 0,1`、`--device cpu`。

### Transformer 融合训练

使用双流 Transformer 融合模型：
```bash
python train_dual.py \
  --model ultralytics/cfg/models/26/yolo26-dual-transformer.yaml \
  --data your_dual_dataset.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

该模型会将 6 通道输入拆分为 RGB/IR 两个 3 通道流，并在多个尺度使用 GPT 模块进行特征融合。

---

## 推理使用

```bash
python detect_dual.py \
  --weights runs/train/dual_stream_exp/weights/best.pt \
  --rgb-source test_images/rgb/ \
  --ir-source test_images/ir/ \
  --save-dir runs/detect/dual_stream_exp \
  --device 0
```

如果使用 Transformer 融合模型，建议显式传入 `--model`：
```bash
python detect_dual.py \
  --weights runs/train/dual_stream_exp/weights/best.pt \
  --model ultralytics/cfg/models/26/yolo26-dual-transformer.yaml \
  --rgb-source test_images/rgb/ \
  --ir-source test_images/ir/ \
  --save-dir runs/detect/dual_stream_exp \
  --device 0
```

- `--rgb-source` 与 `--ir-source` 必须同为文件或文件夹。
- 文件夹模式下按文件名配对，建议 RGB/IR 文件名严格一致。

---

## 快速测试

```bash
python test_dual_stream.py
```

---

## 说明与建议

- 6 通道输入会增加显存占用，建议适当下调 `--batch` 或 `--imgsz`。
- 若发现 IR 与 RGB 尺寸不一致，系统会自动缩放 IR 以匹配 RGB。
- 训练配置与原版 YOLOv26 基本一致，可直接复用原有超参数设置。
