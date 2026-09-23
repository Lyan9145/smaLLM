"""Causal top-1 mixture-of-experts decoder.

Routing is deterministic argmax routing with no token dropping. Every token is
processed by exactly one expert, and the model exposes the same stateless
forward/predict_log_probs interface as the regular student model.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from student_model import Attention, RMSNorm


class Top1MoE(nn.Module):
    def __init__(self, width, hidden_width, num_experts, dropout=0.0,
                 aux_weight=0.01):
        super().__init__()
        if num_experts < 2:
            raise ValueError('num_experts must be at least 2')
        self.width = width
        self.num_experts = num_experts
        self.aux_weight = float(aux_weight)
        self.router = nn.Linear(width, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                'up': nn.Linear(width, 2 * hidden_width, bias=False),
                'down': nn.Linear(hidden_width, width, bias=False),
            }) for _ in range(num_experts)
        ])
        self.dropout = nn.Dropout(dropout)
        self.aux_loss = torch.zeros(())
        self.last_routes = None

    def forward(self, x):
        batch, length, width = x.shape
        flat = x.reshape(-1, width)
        router_logits = self.router(flat)
        probabilities = F.softmax(router_logits.float(), dim=-1)
        routes = probabilities.argmax(dim=-1)
        fractions = F.one_hot(routes, self.num_experts).float().mean(dim=0)
        importance = probabilities.mean(dim=0)
        self.aux_loss = self.aux_weight * self.num_experts * (fractions * importance).sum()
        self.last_routes = routes.detach()

        output = flat.new_zeros(flat.shape)
        for expert_id, expert in enumerate(self.experts):
            indices = (routes == expert_id).nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            selected = flat.index_select(0, indices)
            gate, value = expert['up'](selected).chunk(2, dim=-1)
            selected_output = self.dropout(expert['down'](F.silu(gate) * value))
            routing_weight = probabilities.index_select(0, indices)[:, expert_id]
            selected_output = selected_output * routing_weight.to(selected_output.dtype)[:, None]
            # CUDA autocast can leave the expert result in a different dtype
            # from the destination buffer (for example float32 router/math
            # mixed with bf16 activations).  ``scatter`` requires both dtypes
            # to match, so normalize the routed result before copying it back.
            output.index_copy_(0, indices, selected_output.to(dtype=output.dtype))
        return output.reshape(batch, length, width)


class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config['width']
        hidden = config.get('expert_hidden_width', config.get('hidden_width', math.ceil((8 * width / 3) / 16) * 16))
        dropout = config.get('dropout', 0.0)
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.attention = Attention(width, config['heads'], config['context'], dropout)
        self.moe = Top1MoE(width, hidden, config.get('num_experts', 2), dropout,
                           config.get('moe_aux_weight', 0.01))
        self.dropout = nn.Dropout(dropout)

    @property
    def aux_loss(self):
        return self.moe.aux_loss

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.dropout(self.moe(self.norm2(x)))


class MoEStudentLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width, heads = config['width'], config['heads']
        if width <= 0 or heads <= 0 or width % heads or (width // heads) % 2:
            raise ValueError('width must divide into even-sized attention heads')
        if config['depth'] < 1 or not 0 <= config.get('dropout', 0.0) < 1:
            raise ValueError('depth must be positive and dropout in [0, 1)')
        if config.get('num_experts', 2) < 2:
            raise ValueError('num_experts must be at least 2')
        self.token = nn.Embedding(config['vocab'], width)
        self.blocks = nn.ModuleList([MoEBlock(config) for _ in range(config['depth'])])
        self.norm = RMSNorm(width)
        self.aux_loss = torch.zeros(())
        self.apply(self.initialize)
        for block in self.blocks:
            nn.init.normal_(block.attention.proj.weight, std=0.02 / math.sqrt(2 * config['depth']))
            for expert in block.moe.experts:
                nn.init.normal_(expert['down'].weight, std=0.02 / math.sqrt(2 * config['depth']))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.context:
            raise ValueError('ids must have shape [batch, time] with 1 <= time <= context')
        x = self.token(ids)
        losses = []
        for block in self.blocks:
            x = block(x)
            losses.append(block.aux_loss)
        self.aux_loss = torch.stack(losses).mean() if losses else x.new_zeros(())
        return F.linear(self.norm(x), self.token.weight)

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)
