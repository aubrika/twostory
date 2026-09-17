"""TwoStory = frozen Qwen2.5-7B context tower + Dream-7B denoiser with bridges (+ LoRA).

Two decoding modes:
- `generate_full`: Dream's OWN diffusion loop over the whole padded sequence, bridges live, cache present.
  With a zero-init bridge this must be token-identical to plain Dream (Phase 1 regression gate).
- `generate_block`: the TwoTower loop. Denoiser sees only [anchor, B masks]; the prompt reaches it solely
  through the bridge. Emits per-block timing for the benchmark harness (TTFT, per-block latency)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .bridge import BridgeConfig, attach_bridges, bridge_parameters
from .context_tower import ContextTower
from .dream import DreamModel, sample_tokens
from .util import DEVICE, DREAM_ID, DREAM_REV, DENOISER_BITS, CONTEXT_BITS, _load_kwargs, load_qwen

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class LoraSpec:
    rank: int = 16
    alpha: int = 32
    targets: tuple = tuple(LORA_TARGETS)


def load_dream_vendored(bits: int = DENOISER_BITS):
    m = DreamModel.from_pretrained(DREAM_ID, revision=DREAM_REV, **_load_kwargs(bits))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class TwoStory:
    def __init__(self, dream, qwen, bridge_cfg: BridgeConfig | None = None, lora: LoraSpec | None = None):
        self.dream = dream
        self.ctx = ContextTower(qwen)
        self.mask_id = dream.config.mask_token_id
        self.eos_id = dream.config.eos_token_id
        self.bridges = attach_bridges(dream, bridge_cfg or BridgeConfig(), DEVICE)
        self.lora_params = []
        if lora is not None:
            from peft import LoraConfig, inject_adapter_in_model

            cfg = LoraConfig(r=lora.rank, lora_alpha=lora.alpha, target_modules=list(lora.targets), lora_dropout=0.0, bias="none")
            inject_adapter_in_model(cfg, dream)  # in place; LoRA B is zero-init -> exact no-op at step 0
            self.lora_params = [p for n, p in dream.named_parameters() if "lora_" in n and ".bridge." not in n]
            for p in self.lora_params:
                p.requires_grad_(True)
        # peft's injection freezes every non-adapter parameter, including the bridges: re-enable them last.
        for p in bridge_parameters(dream):
            p.requires_grad_(True)

    @classmethod
    def load(cls, bridge_cfg=None, lora=None, denoiser_bits=DENOISER_BITS, context_bits=CONTEXT_BITS):
        qwen = load_qwen(context_bits)
        dream = load_dream_vendored(denoiser_bits)
        return cls(dream, qwen, bridge_cfg, lora)

    # ------------------------------------------------------------------ parameters
    def trainable_parameters(self):
        return bridge_parameters(self.dream) + self.lora_params

    def param_report(self) -> dict:
        n_bridge = sum(p.numel() for p in bridge_parameters(self.dream))
        n_lora = sum(p.numel() for p in self.lora_params)
        n_dream = sum(p.numel() for p in self.dream.parameters())
        n_qwen = sum(p.numel() for p in self.ctx.qwen.parameters())
        n_train = sum(p.numel() for p in self.dream.parameters() if p.requires_grad) + \
                  sum(p.numel() for p in self.ctx.qwen.parameters() if p.requires_grad)
        return {"bridge": n_bridge, "lora": n_lora, "trainable_total": n_train, "dream_total": n_dream, "qwen_total": n_qwen}

    def train(self, mode=True):
        self.dream.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # ------------------------------------------------------------------ full mode (regression)
    @torch.no_grad()
    def generate_full(self, prompt_ids, **gen_kwargs):
        """Dream's own loop with the bridge live. prompt_ids are Dream-style (BOS-prefixed) and are fed to
        the context tower unchanged so positions align 1:1 between towers."""
        self.ctx.build(prompt_ids)
        self.dream._ambient_ctx = (self.ctx.kv(), None)
        try:
            return self.dream.diffusion_generate(prompt_ids, **gen_kwargs)
        finally:
            self.dream._ambient_ctx = (None, None)

    # ------------------------------------------------------------------ block mode (TwoTower)
    @torch.no_grad()
    def generate_block(self, prompt_ids, max_new_tokens=256, block_size=32, steps=8, alg="entropy",
                       temperature=0.0, top_p=None, top_k=None, eps=1e-3, stop_at_eos=True, on_block=None):
        """prompt_ids: [1, T] Qwen-style (no BOS). Returns dict with ids, timing, and per-block records."""
        assert prompt_ids.shape[0] == 1, "block mode is single-sequence"
        dev = prompt_ids.device
        t0 = time.perf_counter()
        self.ctx.build(prompt_ids)
        torch.xpu.synchronize()
        t_prefill = time.perf_counter() - t0
        committed = prompt_ids
        new_tokens, blocks, ttft = [], [], None
        while len(new_tokens) < max_new_tokens:
            B = min(block_size, max_new_tokens - len(new_tokens))
            tb = time.perf_counter()
            T = self.ctx.length
            x = torch.cat([committed[:, -1:], torch.full((1, B), self.mask_id, device=dev, dtype=torch.long)], dim=1)
            pos = torch.arange(T - 1, T + B, device=dev).unsqueeze(0)
            kv = self.ctx.kv()
            x = self._denoise_block(x, pos, kv, steps, alg, temperature, top_p, top_k, eps)
            block = x[:, 1:]
            torch.xpu.synchronize()
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            blocks.append({"start": T, "size": B, "seconds": now - tb})
            toks = block[0].tolist()
            if stop_at_eos and self.eos_id in toks:
                toks = toks[: toks.index(self.eos_id) + 1]
                block = block[:, : len(toks)]
            new_tokens += toks
            committed = torch.cat([committed, block], dim=1)
            if stop_at_eos and toks and toks[-1] == self.eos_id:
                break
            if len(new_tokens) < max_new_tokens:
                self.ctx.extend(block)
            if on_block:
                on_block(blocks[-1])
        total = time.perf_counter() - t0
        return {"ids": new_tokens, "prefill_seconds": t_prefill, "ttft_seconds": ttft, "total_seconds": total,
                "gen_seconds": total - ttft + blocks[0]["seconds"], "blocks": blocks, "block_size": block_size, "steps": steps}

    def _denoise_block(self, x, pos, kv, steps, alg, temperature, top_p, top_k, eps):
        """Dream's _sample schedule applied to one block (index 0 is the clean anchor, never masked)."""
        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
        for i in range(steps):
            mask_index = x == self.mask_id
            if not mask_index.any():
                break
            logits = self.dream(x, position_ids=pos, ctx_kv=kv).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)  # Dream's shift: hidden i predicts token i+1
            mask_logits = logits[mask_index]
            t, s = timesteps[i], timesteps[i + 1]
            if alg == "origin":
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.full_like(x[mask_index], self.mask_id)
                transfer = torch.rand(*x0.shape, device=x.device) < p_transfer
                _, x0[transfer] = sample_tokens(mask_logits[transfer], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0
            else:
                kw = {"maskgit_plus": {}, "topk_margin": {"margin_confidence": True}, "entropy": {"neg_entropy": True}}[alg]
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, **kw)
                num_mask = mask_index.sum() / mask_index.shape[0]
                n_transfer = int(num_mask * (1 - s / t)) if i < steps - 1 else int(num_mask)
                full_conf = torch.full_like(x, -torch.inf, dtype=logits.dtype)
                full_conf[mask_index] = confidence
                if n_transfer > 0:
                    _, idx = torch.topk(full_conf, n_transfer)
                    x_ = torch.full_like(x, self.mask_id); x_[mask_index] = x0
                    rows = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(idx)
                    x[rows, idx] = x_[rows, idx]
        return x

    # ------------------------------------------------------------------ training forward
    def training_loss(self, seq: torch.LongTensor, ctx_len: int, block_size: int, t: torch.Tensor | None = None,
                      eps: float = 1e-3, weight_by_t: bool = True):
        """One packed sequence [1, L] -> masked-block diffusion loss with the prompt supplied via the cache.
        Blocks j cover [ctx_len + jB, ctx_len + (j+1)B); block j cross-attends to cache positions < its start
        (causal over context), self-attends bidirectionally within [anchor, block]. Loss on masked targets only."""
        L = seq.shape[1]
        assert (L - ctx_len) % block_size == 0 and ctx_len >= 1
        n = (L - ctx_len) // block_size
        dev = seq.device
        with torch.no_grad():
            self.ctx.build(seq)
            kv = self.ctx.kv()
        starts = ctx_len + block_size * torch.arange(n, device=dev)                       # [n]
        targets = seq[0][starts[:, None] + torch.arange(block_size, device=dev)]          # [n, B]
        anchors = seq[0][starts - 1][:, None]                                             # [n, 1]
        if t is None:
            t = torch.rand(n, device=dev).clamp_(eps, 1.0)
        masked = torch.rand(n, block_size, device=dev) < t[:, None]
        # ensure at least one masked token per block
        none = ~masked.any(1)
        masked[none, torch.randint(0, block_size, (int(none.sum()),), device=dev)] = True
        x = torch.cat([anchors, torch.where(masked, torch.full_like(targets, self.mask_id), targets)], dim=1)  # [n, B+1]
        pos = (starts - 1)[:, None] + torch.arange(block_size + 1, device=dev)[None]      # [n, B+1]
        key_pos = torch.arange(L, device=dev)
        ctx_mask = (key_pos[None, None, None, :] < starts[:, None, None, None]).expand(n, 1, block_size + 1, L)
        logits = self.dream(x, position_ids=pos, ctx_kv=kv, ctx_mask=ctx_mask).logits          # [n, B+1, V]
        pred = logits[:, :-1].float()                                                    # hidden i predicts token i+1
        ce = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), targets.reshape(-1), reduction="none").view(n, block_size)
        w = (1.0 / t)[:, None] if weight_by_t else torch.ones_like(t)[:, None]
        loss = (ce * masked * w).sum() / (masked.sum() * (w.mean() if weight_by_t else 1.0))
        return loss, {"n_blocks": n, "n_masked": int(masked.sum()), "ce_masked_mean": ((ce.detach() * masked).sum() / masked.sum()).item()}
