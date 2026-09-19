"""Phase 5 benchmark harness (docs/BENCHMARK.md). Raw per-run records to JSONL; aggregate separately (bench/analyze.py).
Systems: Q (Qwen2.5-7B bf16 greedy AR), D (plain Dream, LoRA off, no bridge), T0 (zero bridge, LoRA off), T (trained),
Tbf16 (trained bridge + bf16 context tower). Decoding is deterministic everywhere (greedy / temperature 0)."""
from __future__ import annotations

import json, math, statistics, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from twostory.util import REPO, DREAM_ID, DREAM_REV, QWEN_ID, QWEN_REV, gib, gpu_telemetry, gpu_other_users, load_tokenizer, provenance
from twostory.prompts import PROMPT_SHORT

BOS_DREAM = 151665
PROMPT_SET = REPO / "data/prompt_set.json"


def load_prompt_set(tiers=None, limit_per_tier=None):
    ps = json.load(open(PROMPT_SET))
    prompts = [p for p in ps["prompts"] if tiers is None or p["tier"] in tiers]
    if limit_per_tier:
        out, seen = [], {}
        for p in prompts:
            if seen.get(p["tier"], 0) < limit_per_tier:
                out.append(p); seen[p["tier"]] = seen.get(p["tier"], 0) + 1
        prompts = out
    return prompts, ps


class Recorder:
    """Append-only JSONL with resume-by-key."""

    def __init__(self, path: Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()
        if self.path.exists():
            for l in open(self.path):
                try:
                    self.done.add(json.loads(l)["key"])
                except Exception:
                    pass
        self.f = open(self.path, "a")

    def has(self, key): return key in self.done

    def write(self, rec):
        rec["time"] = time.time(); self.f.write(json.dumps(rec, default=str) + "\n"); self.f.flush(); self.done.add(rec["key"])


def timed(fn):
    torch.xpu.synchronize(); torch.xpu.reset_peak_memory_stats(); t0 = time.perf_counter()
    out = fn()
    torch.xpu.synchronize()
    return out, time.perf_counter() - t0, gib(torch.xpu.max_memory_allocated())


def _judge_pair(logits, gen_ids):
    """(mean NLL over all generated tokens, mean NLL over tokens that are NOT immediate repeats). Repetition is trivially
    predictable for an AR judge, so the plain NLL rewards degenerate text; the no-repeat variant is the one to rank by."""
    nll = F.cross_entropy(logits, torch.tensor(gen_ids, device="xpu"), reduction="none")
    keep = torch.tensor([not (k >= 2 and (gen_ids[k] == gen_ids[k - 1] or gen_ids[k] == gen_ids[k - 2])) for k in range(len(gen_ids))], device="xpu")
    return {"all": float(nll.mean()), "norep": float(nll[keep].mean()) if keep.any() else float("nan"), "n_norep": int(keep.sum())}


def rep_rate(ids):
    return sum(1 for k in range(2, len(ids)) if ids[k] == ids[k - 1] or ids[k] == ids[k - 2]) / max(1, len(ids) - 2)


# =============================================================================== main session: D / T0 / T (+ Tbf16)
class MainSession:
    def __init__(self, ckpt: str, context_bits: int = 8, bridge_dtype=torch.bfloat16):
        from twostory.model import TwoStory, LoraSpec
        from twostory.bridge import BridgeConfig, bridge_state_dict
        from twostory.train import Trainer, TrainConfig

        self.ts = TwoStory.load(BridgeConfig(), LoraSpec(rank=16), context_bits=context_bits, keep_lm_head=True)
        self.tr = Trainer(self.ts, TrainConfig(run_name="bench_tmp", grad_checkpointing=False))
        self.tr.load_trainable(REPO / ckpt, load_optimizer=False)
        self.ts.eval(); self.ts.bridges_to(bridge_dtype)
        self.trained = {k: v.detach().clone() for k, v in bridge_state_dict(self.ts.dream).items()}
        self.tokq, self.tokd = load_tokenizer("qwen"), load_tokenizer("dream")
        self.system = None
        self.ctx_bits = context_bits

    # ---- system switching (no reload)
    def _lora(self, on):
        from peft.tuners.tuners_utils import BaseTunerLayer
        for m in self.ts.dream.modules():
            if isinstance(m, BaseTunerLayer): m.enable_adapters(on)

    def _bridges(self, trained: bool):
        layers = self.ts.dream.model.layers
        with torch.no_grad():
            for k, v in self.trained.items():
                _, i, _, *rest = k.split(".")
                dst = layers[int(i)].bridge.state_dict()[".".join(rest)]
                dst.copy_(v if trained else torch.zeros_like(v))

    def set_system(self, name):
        if name == self.system: return
        if name == "D": self._lora(False); self.ts.dream._ambient_ctx = (None, None)
        elif name == "T0": self._lora(False); self._bridges(False)
        elif name in ("T", "Tbf16"): self._lora(True); self._bridges(True)
        else: raise ValueError(name)
        self.system = name

    # ---- generation
    @torch.no_grad()
    def gen_D(self, ids, new_tokens, S):
        x = torch.tensor([[BOS_DREAM] + ids], device="xpu")
        out, total, peak = timed(lambda: self.ts.dream.diffusion_generate(x, max_new_tokens=new_tokens, steps=S, temperature=0.0, alg="entropy", alg_temp=None))
        gen = out[0, x.shape[1]:].tolist()
        # Dream has no streaming: every step touches the whole sequence. Prompt cost per step = one forward; report it as TTFT proxy.
        return {"ids": gen, "total_s": total, "ttft_s": total / S, "prefill_s": total / S, "steady_tok_s": new_tokens / total, "peak_gib": peak, "forwards": S}

    @torch.no_grad()
    def gen_T(self, ids, new_tokens, S, B):
        x = torch.tensor([ids], device="xpu")
        r, total, peak = timed(lambda: self.ts.generate_block(x, max_new_tokens=new_tokens, block_size=B, steps=S, stop_at_eos=False))
        first = r["blocks"][0]["size"]
        steady = (new_tokens - first) / max(1e-6, r["total_seconds"] - r["ttft_seconds"]) if new_tokens > first else new_tokens / r["total_seconds"]
        return {"ids": r["ids"], "total_s": r["total_seconds"], "ttft_s": r["ttft_seconds"], "prefill_s": r["prefill_seconds"], "steady_tok_s": steady,
                "per_block_s": [round(b["seconds"], 4) for b in r["blocks"]], "peak_gib": peak, "forwards": S * len(r["blocks"])}

    # ---- judge: NLL of generated tokens under the (int8) Qwen with its lm_head, conditional on the prompt
    @torch.no_grad()
    def judge_nll(self, prompt_ids, gen_ids):
        x = torch.tensor([prompt_ids + gen_ids], device="xpu")
        logits = self.ts.ctx.qwen(x).logits[0, len(prompt_ids) - 1:-1].float()
        return _judge_pair(logits, gen_ids)

    # ---- held-out masked CE (same 32 sequences/masks as phase3/32_reference_loss.py) for T at block size B
    @torch.no_grad()
    def heldout_ce_T(self, B=32, n_seqs=32, ctx=512):
        tot, cnt = 0.0, 0
        for i in range(n_seqs):
            g = torch.Generator(device="xpu"); g.manual_seed(10_000 + i)
            _, info = self.ts.training_loss(self.tr.seq(self.tr.held, i), ctx, B, generator=g)
            tot += info["ce_masked_mean"] * info["n_masked"]; cnt += info["n_masked"]
        return tot / cnt

    @torch.no_grad()
    def longctx_ce(self, system, ctx_len, n_seqs=16, B=32, n_blocks=4):
        """Masked CE on n_blocks blocks following ctx_len tokens of held-out text (consecutive held-out sequences concatenated
        for ctx > 1024). D sees the context in-input; T via the bridge. Same masks (seeded) for both."""
        self.set_system(system); L = self.tr.L; need = ctx_len + n_blocks * B; k = -(-need // L)
        tot, cnt = 0.0, 0
        for i in range(n_seqs):
            seq = torch.from_numpy(self.tr.held[i * k:(i + 1) * k].astype(np.int64).reshape(-1)[:need]).to("xpu")
            g = torch.Generator(device="xpu"); g.manual_seed(20_000 + i)
            t = torch.rand(n_blocks, device="xpu", generator=g).clamp_(1e-3, 1.0)
            masked = torch.rand(n_blocks, B, device="xpu", generator=g) < t[:, None]
            none = ~masked.any(1); masked[none, 0] = True
            for j in range(n_blocks):
                s = ctx_len + j * B; tgt = seq[s:s + B]; blk = torch.where(masked[j], torch.full((B,), self.ts.mask_id, device="xpu"), tgt)
                if system == "D":
                    x = torch.cat([torch.tensor([BOS_DREAM], device="xpu"), seq[:s], blk])[None]
                    pred = self.ts.dream(x).logits[0, s: s + B].float()          # with BOS offset: hidden (p-1)+1 predicts p
                else:
                    self.ts.ctx.build(seq[:s][None]); kv = self.ts.ctx.kv()
                    x = torch.cat([seq[s - 1: s], blk])[None]; pos = torch.arange(s - 1, s + B, device="xpu")[None]
                    hidden = self.ts.dream.model(input_ids=x, position_ids=pos, ctx_kv=kv, use_cache=False).last_hidden_state
                    pred = self.ts.dream.lm_head(hidden[0, :-1]).float()
                ce = F.cross_entropy(pred, tgt, reduction="none")
                tot += float(ce[masked[j]].sum()); cnt += int(masked[j].sum())
        return tot / cnt


# =============================================================================== Q session
class QSession:
    def __init__(self):
        from twostory.util import load_qwen
        self.qwen = load_qwen(16); self.tokq = load_tokenizer("qwen")

    @torch.no_grad()
    def gen(self, ids, new_tokens):
        x = torch.tensor([ids], device="xpu")
        _, prefill, _ = timed(lambda: self.qwen.generate(x, max_new_tokens=1, min_new_tokens=1, do_sample=False, pad_token_id=self.tokq.eos_token_id))
        out, total, peak = timed(lambda: self.qwen.generate(x, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, pad_token_id=self.tokq.eos_token_id))
        gen = out[0, x.shape[1]:].tolist()
        return {"ids": gen, "total_s": total, "ttft_s": prefill, "prefill_s": prefill, "steady_tok_s": (new_tokens - 1) / max(1e-6, total - prefill), "peak_gib": peak, "forwards": new_tokens}

    @torch.no_grad()
    def judge_nll(self, prompt_ids, gen_ids):
        x = torch.tensor([prompt_ids + gen_ids], device="xpu")
        logits = self.qwen(x).logits[0, len(prompt_ids) - 1:-1].float()
        return _judge_pair(logits, gen_ids)

    @torch.no_grad()
    def heldout_ppl(self, n_seqs=32):
        held = np.memmap(REPO / "data/tokens/heldout.bin", dtype=np.uint32, mode="r").reshape(-1, 1024)
        tot, cnt = 0.0, 0
        for i in range(n_seqs):
            x = torch.from_numpy(held[i].astype(np.int64))[None].to("xpu")
            logits = self.qwen(x).logits[0, :-1].float()
            tot += float(F.cross_entropy(logits, x[0, 1:], reduction="sum")); cnt += x.shape[1] - 1
        return {"nll_per_token": tot / cnt, "ppl": math.exp(tot / cnt), "n_tokens": cnt}


def session_header(name, extra=None):
    h = {"session": name, "provenance": provenance(), "gpu_other_users": gpu_other_users(), "gpu": gpu_telemetry(), "prompt_set_sha256": open(REPO / "data/prompt_set.sha256").read().strip()}
    if extra: h.update(extra)
    return h
