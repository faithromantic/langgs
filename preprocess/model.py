
import math
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(dims: List[int], last_activation: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        in_dim = dims[i]
        out_dim = dims[i + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        is_last = (i == len(dims) - 2)
        if (not is_last) or last_activation:
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class FeatureAutoEncoder(nn.Module):
    """
    为 LangSplat -> Scaffold-GS 的预处理设计：
    - 输入: CLIP 512-d feature
    - 输出: latent_dim-d 压缩语义
    - decoder 用于重建训练，保证 latent 保持语义可逆
    """
    def __init__(
        self,
        input_dim: int = 512,
        latent_dim: int = 3,
        encoder_hidden: List[int] = None,
        decoder_hidden: List[int] = None,
        normalize_input: bool = True,
        normalize_latent: bool = False,
    ):
        super().__init__()
        encoder_hidden = encoder_hidden or [256, 128, 32]
        decoder_hidden = decoder_hidden or [32, 128, 256]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.normalize_input = normalize_input
        self.normalize_latent = normalize_latent

        self.encoder = build_mlp([input_dim] + encoder_hidden + [latent_dim], last_activation=False)
        self.decoder = build_mlp([latent_dim] + decoder_hidden + [input_dim], last_activation=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x = F.normalize(x, dim=-1)
        z = self.encoder(x)
        if self.normalize_latent:
            z = F.normalize(z, dim=-1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(z)
        x_hat = F.normalize(x_hat, dim=-1)
        return x_hat

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


def reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mse_weight: float = 1.0,
    cosine_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    CLIP 特征更适合用 cosine 约束；MSE 负责数值稳定。
    """
    x = F.normalize(x, dim=-1)
    x_hat = F.normalize(x_hat, dim=-1)

    mse = F.mse_loss(x_hat, x)
    cos = 1.0 - F.cosine_similarity(x_hat, x, dim=-1).mean()
    total = mse_weight * mse + cosine_weight * cos
    return {
        "loss": total,
        "mse": mse,
        "cosine": cos,
    }


def to_config_dict(model: FeatureAutoEncoder) -> Dict[str, Any]:
    return {
        "input_dim": model.input_dim,
        "latent_dim": model.latent_dim,
        "normalize_input": model.normalize_input,
        "normalize_latent": model.normalize_latent,
    }
