"""Cross-attention bridge: denoiser hidden state (queries) -> context tower cache (keys/values), per layer.

Design (docs/PROTOCOL.md §3):
- queries: RMSNorm(hidden) -> the layer's OWN frozen q_proj (shared weight, lineage-aligned with Qwen's key space)
  plus a trainable LoRA delta; RoPE applied at the block's absolute positions so q·k encodes relative offset.
- keys/values: taken directly from the context tower cache (already RoPE'd), GQA-expanded 4 -> 28 heads.
- output: `out_proj`, ZERO-INITIALISED -> the wired model is behaviourally identical to plain Dream at init.
  (Named out_proj, not o_proj, so peft's target_modules for Dream's own attention never touch it.)
Trainable params are kept in fp32; activations are cast in and out."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .dream import DreamRMSNorm, apply_rotary_pos_emb, repeat_kv


@dataclass
class BridgeConfig:
    layers: list[int] = field(default_factory=lambda: list(range(28)))  # which denoiser layers get a bridge
    q_lora_rank: int = 32
    q_lora_alpha: int = 64
    out_rank: int | None = None  # None = full 3584x3584 out_proj; int = low-rank (A: h->r, B: r->h, B zero)
    param_dtype: torch.dtype = torch.float32


class CrossAttentionBridge(nn.Module):
    def __init__(self, config, layer, cfg: BridgeConfig):
        super().__init__()
        h, self.n_heads, self.n_kv = config.hidden_size, config.num_attention_heads, config.num_key_value_heads
        self.head_dim = h // self.n_heads
        self.n_rep = self.n_heads // self.n_kv
        dt = cfg.param_dtype
        # pre-norm, warm-started from the layer's input norm (so the query path sees what self-attn sees)
        self.norm = DreamRMSNorm(h, eps=config.rms_norm_eps)
        self.norm.weight.data.copy_(layer.input_layernorm.weight.data.to(dt))
        # frozen shared query base: reference the layer's q_proj weight/bias WITHOUT registering (no copy, no state_dict)
        object.__setattr__(self, "_q_base", (layer.self_attn.q_proj.weight, layer.self_attn.q_proj.bias))
        self.q_lora_A = nn.Linear(h, cfg.q_lora_rank, bias=False, dtype=dt)
        self.q_lora_B = nn.Linear(cfg.q_lora_rank, h, bias=False, dtype=dt)
        nn.init.kaiming_uniform_(self.q_lora_A.weight, a=math.sqrt(5)); nn.init.zeros_(self.q_lora_B.weight)
        self.q_scale = cfg.q_lora_alpha / cfg.q_lora_rank
        if cfg.out_rank is None:
            self.out_proj = nn.Linear(h, h, bias=False, dtype=dt); nn.init.zeros_(self.out_proj.weight)
            self.out_low = None
        else:
            self.out_low = nn.Linear(h, cfg.out_rank, bias=False, dtype=dt)
            self.out_proj = nn.Linear(cfg.out_rank, h, bias=False, dtype=dt)
            nn.init.kaiming_uniform_(self.out_low.weight, a=math.sqrt(5)); nn.init.zeros_(self.out_proj.weight)
        self.calls = 0  # instrumentation: how many times the bridge actually ran

    def forward(self, hidden, k_ctx, v_ctx, position_embeddings, ctx_mask=None):
        self.calls += 1
        bsz, L, h = hidden.shape
        in_dtype = hidden.dtype
        x = self.norm(hidden.to(self.norm.weight.dtype))
        wq, bq = self._q_base
        q = F.linear(x.to(wq.dtype), wq, bq).to(x.dtype) + self.q_lora_B(self.q_lora_A(x)) * self.q_scale
        q = q.view(bsz, L, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q = (q * cos.unsqueeze(1).to(q.dtype)) + (_rotate_half(q) * sin.unsqueeze(1).to(q.dtype))
        k = repeat_kv(k_ctx, self.n_rep); v = repeat_kv(v_ctx, self.n_rep)
        if k.shape[0] != bsz:
            # One shared cache, many blocks. Do NOT expand K/V across the batch: on XPU, SDPA with stride-0 (expanded)
            # K/V returns NaN under no_grad and picks a pathologically slow kernel (Phase 3 pilot hang, 2026-09-17), and a
            # contiguous copy costs ~350 MB per layer at 24 blocks. Cross-attention is independent per query, so fold the
            # blocks into the query-length dimension instead: q [bsz, H, L, d] -> [1, H, bsz*L, d]; K/V stay [1, H, T, d].
            assert k.shape[0] == 1, "shared-cache path expects a single context sequence"
            qf = q.transpose(0, 1).reshape(1, self.n_heads, bsz * L, self.head_dim)
            mf = None if ctx_mask is None else ctx_mask.expand(bsz, 1, L, k.shape[2]).reshape(1, 1, bsz * L, k.shape[2])
            attn = F.scaled_dot_product_attention(qf.to(k.dtype), k, v, attn_mask=mf, is_causal=False)
            attn = attn.view(self.n_heads, bsz, L, self.head_dim).permute(1, 2, 0, 3).reshape(bsz, L, h).to(x.dtype)
        else:
            if ctx_mask is not None and not ctx_mask.is_contiguous():
                ctx_mask = ctx_mask.contiguous()
            attn = F.scaled_dot_product_attention(q.to(k.dtype), k, v, attn_mask=ctx_mask, is_causal=False)
            attn = attn.transpose(1, 2).reshape(bsz, L, h).to(x.dtype)
        if self.out_low is not None:
            attn = self.out_low(attn)
        return self.out_proj(attn).to(in_dtype)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def attach_bridges(dream, cfg: BridgeConfig, device) -> list[CrossAttentionBridge]:
    """Insert a bridge into each configured DreamDecoderLayer. Returns the bridge modules."""
    bridges = []
    for i in cfg.layers:
        layer = dream.model.layers[i]
        b = CrossAttentionBridge(dream.config, layer, cfg).to(device)
        layer.bridge = b
        bridges.append(b)
    return bridges


def bridge_parameters(dream):
    return [p for l in dream.model.layers if l.bridge is not None for p in l.bridge.parameters()]


def bridge_state_dict(dream) -> dict:
    return {f"layers.{i}.bridge.{k}": v for i, l in enumerate(dream.model.layers) if l.bridge is not None
            for k, v in l.bridge.state_dict().items()}
