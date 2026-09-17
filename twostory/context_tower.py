"""Frozen context tower: Qwen2.5-7B processes clean tokens once, exposing per-layer post-RoPE K/V.
Extend, don't rebuild: `extend()` feeds only the newly committed block through the tower."""
from __future__ import annotations

import torch


def _layer_kv(cache, i):
    if hasattr(cache, "layers"):  # transformers >= 4.56
        return cache.layers[i].keys, cache.layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


class ContextTower:
    def __init__(self, qwen, drop_lm_head: bool = True):
        self.qwen = qwen
        self.base = qwen.model  # Qwen2Model: skip lm_head, the tower never needs logits
        if drop_lm_head and isinstance(getattr(qwen, "lm_head", None), torch.nn.Linear):
            qwen.lm_head = torch.nn.Identity()  # frees 1.09 GiB of bf16 weights the context tower never uses
            import gc; gc.collect(); torch.xpu.empty_cache()
        self.n_layers = qwen.config.num_hidden_layers
        self.cache = None
        self.length = 0

    @torch.no_grad()
    def build(self, ids: torch.LongTensor):
        """ids: [B, T] clean tokens (Qwen style, no BOS). Replaces any existing cache."""
        from transformers import DynamicCache

        self.cache = DynamicCache()
        self.base(input_ids=ids, past_key_values=self.cache, use_cache=True)
        self.length = ids.shape[1]
        return self

    @torch.no_grad()
    def extend(self, new_ids: torch.LongTensor):
        """Append a committed block [B, n] to the cache (processes only the new tokens)."""
        pos = torch.arange(self.length, self.length + new_ids.shape[1], device=new_ids.device).unsqueeze(0)
        self.base(input_ids=new_ids, past_key_values=self.cache, use_cache=True, position_ids=pos, cache_position=pos[0])
        self.length += new_ids.shape[1]
        return self

    def kv(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer (K, V), each [B, n_kv_heads, T, head_dim], RoPE already applied at absolute positions."""
        return [_layer_kv(self.cache, i) for i in range(self.n_layers)]

    def bytes(self) -> int:
        return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in self.kv())
