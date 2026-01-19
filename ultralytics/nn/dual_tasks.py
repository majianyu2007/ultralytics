# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
双流YOLO检测模型

这个模块实现了支持双流输入（RGB + IR）的YOLO检测模型架构
"""

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel, BaseModel, yaml_model_load, attempt_load_weights
from ultralytics.utils import LOGGER
from ultralytics.nn.modules import Conv, Concat


class DualStreamDetectionModel(DetectionModel):
    """
    双流YOLO检测模型

    支持RGB和IR图像的双流输入，内部将6通道输入分离为两个3通道流进行处理，
    然后通过融合机制结合两个流的特征进行最终检测。

    主要特点：
    - 接受6通道输入（RGB 3通道 + IR 3通道）
    - 内部分离为两个独立的3通道处理流
    - 支持多层特征融合
    - 兼容标准YOLO检测头
    """

    def __init__(self, cfg="yolo26n.yaml", ch=6, nc=None, verbose=True):
        """
        初始化双流YOLO检测模型

        Args:
            cfg (str | dict): 模型配置文件或字典
            ch (int): 输入通道数，默认6（RGB 3 + IR 3）
            nc (int): 类别数
            verbose (bool): 是否显示详细信息
        """
        # 强制设置为6通道输入
        if ch != 6:
            LOGGER.warning(f"双流模型要求6通道输入 (RGB 3 + IR 3), 但收到 ch={ch}, 自动设置为6")
            ch = 6

        # 初始化基础检测模型
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # 标记为双流模型
        self.is_dual_stream = True

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        双流预测前向传播

        Args:
            x (torch.Tensor): 输入张量，shape为 (batch_size, 6, height, width)
            profile (bool): 是否性能分析
            visualize (bool): 是否可视化
            augment (bool): 是否数据增强
            embed (list): 特征嵌入层索引

        Returns:
            torch.Tensor: 检测结果
        """
        # 验证输入通道数
        if x.shape[1] != 6:
            raise ValueError(f"双流模型要求6通道输入，但收到 {x.shape[1]} 通道")

        # 分离RGB和IR流
        rgb_stream = x[:, :3, :, :]  # 前3通道 (RGB)
        ir_stream = x[:, 3:, :, :]   # 后3通道 (IR)

        if augment:
            return self._predict_dual_stream_augment(rgb_stream, ir_stream, profile, visualize, embed)
        else:
            return self._predict_dual_stream_once(rgb_stream, ir_stream, profile, visualize, embed)

    def _predict_dual_stream_once(self, rgb_x, ir_x, profile=False, visualize=False, embed=None):
        """
        双流单次前向传播

        Args:
            rgb_x (torch.Tensor): RGB流输入 (batch_size, 3, height, width)
            ir_x (torch.Tensor): IR流输入 (batch_size, 3, height, width)

        Returns:
            torch.Tensor: 网络输出
        """
        y_rgb, y_ir, dt = [], [], []

        # 遍历模型的每一层
        for i, m in enumerate(self.model):
            layer_name = getattr(m, 'layer_name', f'layer_{i}')

            # 根据层的配置决定处理哪个流
            if hasattr(m, 'dual_stream_mode'):
                mode = m.dual_stream_mode
            else:
                # 默认逻辑：根据层的位置和类型判断处理模式
                mode = self._get_layer_mode(i, m)

            if profile:
                from ultralytics.utils.torch_utils import time_sync
                c = m == self.model[-1]  # is final layer
                o = torch.jit.trace(m, (rgb_x.copy() if c else rgb_x,), strict=False)[0].flops / 1E9 * 2 if profile else 0
                t = time_sync()
                for _ in range(10):
                    _ = m(rgb_x.copy() if c else rgb_x)
                dt.append((time_sync() - t) * 100)
                LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            # 根据模式处理流
            if mode == 'rgb_only':
                # 只处理RGB流
                if m.f != -1:  # 不是从上一层
                    rgb_x = y_rgb[m.f] if isinstance(m.f, int) else [rgb_x if j == -1 else y_rgb[j] for j in m.f]
                rgb_x = m(rgb_x)
                y_rgb.append(rgb_x if m.i in self.save else None)

            elif mode == 'ir_only':
                # 只处理IR流
                if m.f != -1:
                    ir_x = y_ir[m.f] if isinstance(m.f, int) else [ir_x if j == -1 else y_ir[j] for j in m.f]
                ir_x = m(ir_x)
                y_ir.append(ir_x if m.i in self.save else None)

            elif mode == 'fusion':
                # 融合两个流
                if hasattr(m, 'forward_dual'):
                    # 自定义双流前向传播
                    fused_output = m.forward_dual(rgb_x, ir_x)
                    if isinstance(fused_output, (list, tuple)):
                        rgb_x, ir_x = fused_output[0], fused_output[1]
                    else:
                        rgb_x = ir_x = fused_output
                else:
                    # 默认融合：简单相加
                    fused = (rgb_x + ir_x) / 2
                    rgb_x = ir_x = fused

                y_rgb.append(rgb_x if m.i in self.save else None)
                y_ir.append(ir_x if m.i in self.save else None)

            elif mode == 'final':
                # 最终层，合并两个流
                if hasattr(m, 'forward_dual'):
                    combined_x = m.forward_dual(rgb_x, ir_x)
                else:
                    # 在通道维度拼接两个流
                    combined_x = torch.cat([rgb_x, ir_x], dim=1)
                    combined_x = m(combined_x)

                y_rgb.append(combined_x if m.i in self.save else None)
                y_ir.append(combined_x if m.i in self.save else None)
                return combined_x

        # 如果没有final层，默认合并输出
        if hasattr(self.model[-1], 'forward_dual'):
            return self.model[-1].forward_dual(rgb_x, ir_x)
        else:
            # 简单融合策略
            return (rgb_x + ir_x) / 2

    def _get_layer_mode(self, layer_idx, module):
        """
        根据层索引和模块类型确定处理模式

        Args:
            layer_idx (int): 层索引
            module (nn.Module): 模块

        Returns:
            str: 处理模式 ('rgb_only', 'ir_only', 'fusion', 'final')
        """
        # 这里可以根据具体的模型架构来定义规则
        # 示例规则：

        # 检测头通常是最后的层
        if isinstance(module, (Detect, YOLOEDetect, v10Detect, Segment, Pose, OBB)):
            return 'final'

        # 前几层分别处理RGB和IR
        if layer_idx < len(self.model) // 4:
            return 'rgb_only' if layer_idx % 2 == 0 else 'ir_only'

        # 中间层进行融合
        elif layer_idx < len(self.model) * 3 // 4:
            return 'fusion'

        # 后面的层继续融合
        else:
            return 'fusion'

    def _predict_dual_stream_augment(self, rgb_x, ir_x, profile=False, visualize=False, embed=None):
        """
        双流增强推理

        Args:
            rgb_x (torch.Tensor): RGB流输入
            ir_x (torch.Tensor): IR流输入

        Returns:
            torch.Tensor: 增强推理结果
        """
        # 实现双流的增强推理
        img_size = rgb_x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs

        for si, fi in zip(s, f):
            # 对RGB和IR应用相同的变换
            xi_rgb = scale_img(rgb_x.flip(fi) if fi else rgb_x, si)
            xi_ir = scale_img(ir_x.flip(fi) if fi else ir_x, si)

            yi = self._predict_dual_stream_once(xi_rgb, xi_ir, profile, visualize, embed)
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)

        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1)


class DualStreamFusion(nn.Module):
    """
    双流融合模块

    可以作为网络中的融合层，将RGB和IR两个流的特征进行融合
    """

    def __init__(self, channels, fusion_type='concat'):
        """
        初始化融合模块

        Args:
            channels (int): 输入通道数
            fusion_type (str): 融合类型 ('concat', 'add', 'attention')
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.channels = channels

        if fusion_type == 'concat':
            self.conv = Conv(channels * 2, channels, 1)
        elif fusion_type == 'attention':
            self.attention = nn.MultiheadAttention(channels, 8)
            self.norm = nn.LayerNorm(channels)

    def forward_dual(self, rgb_feat, ir_feat):
        """
        双流融合前向传播

        Args:
            rgb_feat (torch.Tensor): RGB特征
            ir_feat (torch.Tensor): IR特征

        Returns:
            tuple: 融合后的(rgb_feat, ir_feat)
        """
        if self.fusion_type == 'concat':
            # 拼接融合
            fused = torch.cat([rgb_feat, ir_feat], dim=1)
            fused = self.conv(fused)
            return fused, fused

        elif self.fusion_type == 'add':
            # 相加融合
            fused = (rgb_feat + ir_feat) / 2
            return fused, fused

        elif self.fusion_type == 'attention':
            # 注意力融合
            B, C, H, W = rgb_feat.shape
            rgb_flat = rgb_feat.view(B, C, H*W).permute(2, 0, 1)  # (H*W, B, C)
            ir_flat = ir_feat.view(B, C, H*W).permute(2, 0, 1)    # (H*W, B, C)

            # 交叉注意力
            rgb_attended, _ = self.attention(rgb_flat, ir_flat, ir_flat)
            ir_attended, _ = self.attention(ir_flat, rgb_flat, rgb_flat)

            # 恢复形状
            rgb_out = rgb_attended.permute(1, 2, 0).view(B, C, H, W)
            ir_out = ir_attended.permute(1, 2, 0).view(B, C, H, W)

            # 残差连接
            rgb_out = self.norm(rgb_out.view(B, C, -1)).view(B, C, H, W) + rgb_feat
            ir_out = self.norm(ir_out.view(B, C, -1)).view(B, C, H, W) + ir_feat

            return rgb_out, ir_out

        else:
            raise ValueError(f"不支持的融合类型: {self.fusion_type}")

    def forward(self, x):
        """标准前向传播（用于非双流场景）"""
        return x


# 注册双流融合模块
import sys
from ultralytics.nn.modules import Conv, Concat, Detect
if 'DualStreamFusion' not in globals():
    globals()['DualStreamFusion'] = DualStreamFusion