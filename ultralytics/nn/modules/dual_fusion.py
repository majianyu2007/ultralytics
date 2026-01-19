# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Dual-stream fusion modules."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

__all__ = ("Add", "Add2", "GPT", "SelfAttention")


class Add(nn.Module):
    """Add two tensors."""

    def __init__(self, arg: int | None = None):
        super().__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])


class Add2(nn.Module):
    """Add a tensor with one branch from a tuple output (e.g., GPT)."""

    def __init__(self, c1: int, index: int):
        super().__init__()
        self.index = index

    def forward(self, x):
        return torch.add(x[0], x[1][self.index])


class SelfAttention(nn.Module):
    """Multi-head self-attention used in dual-stream fusion."""

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.que_proj = nn.Linear(d_model, h * self.d_k)
        self.key_proj = nn.Linear(d_model, h * self.d_k)
        self.val_proj = nn.Linear(d_model, h * self.d_v)
        self.out_proj = nn.Linear(h * self.d_v, d_model)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.resid_drop(self.out_proj(out))
        return out


class _TransformerFusionBlock(nn.Module):
    """Transformer block for dual-stream fusion."""

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))
        return x


class GPT(nn.Module):
    """Dual-stream transformer fusion module."""

    def __init__(
        self,
        d_model,
        h=8,
        block_exp=4,
        n_layer=8,
        vert_anchors=8,
        horz_anchors=8,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
    ):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))
        self.trans_blocks = nn.Sequential(
            *[
                _TransformerFusionBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.drop = nn.Dropout(embd_pdrop)
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        rgb_fea, ir_fea = x[0], x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        rgb_fea_flat = rgb_fea.view(bs, c, -1)
        ir_fea_flat = ir_fea.view(bs, c, -1)
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.trans_blocks(x)
        x = self.ln_f(x)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)

        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode="bilinear")
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode="bilinear")

        return rgb_fea_out, ir_fea_out
