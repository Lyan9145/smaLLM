"""Stateless RMSNorm/RoPE/SwiGLU decoder with gated causal local mixing."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        # Accumulate the variance in FP32 even during BF16 training.
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(x.dtype) * self.weight


class CausalLocalMix(nn.Module):
    def __init__(self, width, kernel_size=3):
        super().__init__()
        if kernel_size < 1:
            raise ValueError('local_kernel must be positive')
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(width, width, kernel_size, groups=width, bias=False)
        self.gate = nn.Parameter(torch.full((width,), -2.0))

    def forward(self, x):
        local = self.conv(F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0)))
        return local.transpose(1, 2) * self.gate.sigmoid()


class Attention(nn.Module):
    def __init__(self, width, heads, context, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        head_dim = width // heads
        angles = torch.outer(torch.arange(context).float(),
                             10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def rotate(self, x):
        cos = self.cos[:x.shape[-2]].to(x.dtype)
        sin = self.sin[:x.shape[-2]].to(x.dtype)
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v,
                                            is_causal=True,
                                            dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config['width']
        hidden = config.get('hidden_width', math.ceil((8 * width / 3) / 16) * 16)
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.attention = Attention(width, config['heads'], config['context'], config.get('dropout', 0.0))
        self.up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(config.get('dropout', 0.0))
        self.local = CausalLocalMix(width, config.get('local_kernel', 3)) if config.get('local_mixing', True) else None

    def forward(self, x):
        normalized = self.norm1(x)
        mixed = self.attention(normalized)
        if self.local is not None:
            mixed = mixed + self.local(normalized)
        x = x + self.dropout(mixed)
        gate, value = self.up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.dropout(self.down(F.silu(gate) * value))


class StudentLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width, heads = config['width'], config['heads']
        if width <= 0 or heads <= 0 or width % heads or (width // heads) % 2:
            raise ValueError('width must divide into even-sized attention heads')
        if config['depth'] < 1 or not 0 <= config.get('dropout', 0.0) < 1:
            raise ValueError('depth must be positive and dropout in [0, 1)')
        self.token = nn.Embedding(config['vocab'], width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['depth'])])
        self.norm = RMSNorm(width)
        self.apply(self.initialize)
        for block in self.blocks:
            nn.init.normal_(block.attention.proj.weight, std=0.02 / math.sqrt(2 * config['depth']))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * config['depth']))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.context:
            raise ValueError('ids must have shape [batch, time] with 1 <= time <= context')
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.norm(x), self.token.weight)

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)
