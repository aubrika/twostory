"""Training loop for the bridge + LoRA. Used by Phase 2 (overfit), Phase 3 (pilot) and Phase 4 (the run).
Raw per-step records go to a JSONL log; checkpoints hold only trainable state (bridge + LoRA) plus optimizer."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

from .bridge import bridge_state_dict
from .util import REPO, gpu_telemetry, provenance, gib


@dataclass
class TrainConfig:
    steps: int = 100
    block_size: int = 32
    ctx_min: int = 256
    ctx_max: int = 768
    batch_seqs: int = 1            # sequences per optimizer step (sequential, gradients accumulated)
    lr: float = 1e-4
    min_lr_ratio: float = 0.1
    warmup: int = 50
    weight_decay: float = 0.0
    clip: float = 1.0
    optimizer: str = "adamw8bit"   # adamw | adamw8bit (8-bit is the Phase 2 decision: full bridge fits under 28 GiB)
    seed: int = 0
    eval_every: int = 0            # 0 = never
    eval_seqs: int = 32
    eval_ctx: int = 512
    ckpt_every_sec: float = 900         # 15 min: a driver wedge costs at most this much progress
    keep_best: bool = True
    early_stop_patience: int = 3   # consecutive eval increases before stopping (0 = off)
    log_every: int = 1
    run_name: str = "run"
    weight_by_t: bool = True
    grad_checkpointing: bool = True


class Trainer:
    def __init__(self, ts, cfg: TrainConfig, run_dir: Path | None = None):
        self.ts, self.cfg = ts, cfg
        self.run_dir = Path(run_dir or REPO / "checkpoints" / cfg.run_name)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        meta = json.load(open(REPO / "data/tokens/meta.json"))
        self.L = meta["seq_len"]
        self.train = np.memmap(REPO / "data/tokens/train.bin", dtype=np.uint32, mode="r").reshape(-1, self.L)
        self.held = np.memmap(REPO / "data/tokens/heldout.bin", dtype=np.uint32, mode="r").reshape(-1, self.L)
        self.rng = np.random.default_rng(cfg.seed)
        self.gen = torch.Generator(device="xpu"); self.gen.manual_seed(cfg.seed)
        self.params = ts.trainable_parameters()
        self.opt = self._make_optimizer()
        self.step = 0
        self.tokens_seen = 0
        self.best_eval = math.inf
        self.eval_history: list[float] = []
        self.log_path = self.run_dir / "train.jsonl"
        self.log_f = open(self.log_path, "a")
        (self.run_dir / "config.json").write_text(json.dumps({"train": asdict(cfg), "params": ts.param_report(),
                                                                 "provenance": provenance()}, indent=2, default=str))
        if cfg.grad_checkpointing:
            ts.dream.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def _make_optimizer(self):
        c = self.cfg
        if c.optimizer == "adamw":
            return torch.optim.AdamW(self.params, lr=c.lr, weight_decay=c.weight_decay, betas=(0.9, 0.95), foreach=False)
        if c.optimizer == "adamw8bit":
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(self.params, lr=c.lr, weight_decay=c.weight_decay, betas=(0.9, 0.95))
        raise ValueError(c.optimizer)

    def lr_at(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup:
            return c.lr * (step + 1) / c.warmup
        p = min(1.0, (step - c.warmup) / max(1, c.steps - c.warmup))
        return c.lr * (c.min_lr_ratio + (1 - c.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))

    def sample_ctx_len(self) -> int:
        B = self.cfg.block_size
        return B * int(self.rng.integers(self.cfg.ctx_min // B, self.cfg.ctx_max // B + 1))

    def seq(self, arr, i) -> torch.Tensor:
        return torch.from_numpy(arr[i].astype(np.int64)).unsqueeze(0).to("xpu")

    # ------------------------------------------------------------------ one optimizer step
    def train_step(self, seq_idx: list[int] | None = None, ctx_len: int | None = None, reseed: int | None = None) -> dict:
        c = self.cfg
        self.ts.train()
        if seq_idx is None:
            seq_idx = self.rng.integers(0, len(self.train), size=c.batch_seqs).tolist()
        if reseed is not None:
            self.gen.manual_seed(reseed)
        lr = self.lr_at(self.step)
        for g in self.opt.param_groups:
            g["lr"] = lr
        t0 = time.perf_counter()
        self.opt.zero_grad(set_to_none=True)
        losses, n_masked = [], 0
        for i in seq_idx:
            cl = ctx_len or self.sample_ctx_len()
            loss, info = self.ts.training_loss(self.seq(self.train, i), cl, c.block_size, weight_by_t=c.weight_by_t, generator=self.gen)
            (loss / len(seq_idx)).backward()
            losses.append(loss.item()); n_masked += info["n_masked"]
        gn = torch.nn.utils.clip_grad_norm_(self.params, c.clip).item() if c.clip else float("nan")
        self.opt.step()
        torch.xpu.synchronize()
        dt = time.perf_counter() - t0
        self.step += 1
        self.tokens_seen += len(seq_idx) * self.L
        # Allocator guard: varying block counts fragment the caching allocator (29.4 GiB reserved vs 26.0 allocated in
        # the Phase 3 diagnostic). Running the device out of VRAM makes kernels crawl past xe's 5 s job timeout and
        # wedges the process, so hand cached blocks back whenever the slack gets large.
        if torch.xpu.memory_reserved() - torch.xpu.memory_allocated() > 1.5 * 2**30:
            torch.xpu.empty_cache()
        dev_free = torch.xpu.mem_get_info()[0]
        rec = {"step": self.step, "loss": float(np.mean(losses)), "lr": lr, "grad_norm": gn, "seconds": round(dt, 3),
               "seq_tok_s": round(len(seq_idx) * self.L / dt, 1), "n_masked": n_masked, "tokens_seen": self.tokens_seen,
               "vram_alloc_gib": gib(torch.xpu.memory_allocated()), "vram_peak_gib": gib(torch.xpu.max_memory_allocated()),
               "vram_reserved_gib": gib(torch.xpu.memory_reserved()), "dev_free_gib": gib(dev_free), "time": time.time()}
        if self.step % c.log_every == 0:
            rec["gpu"] = {k: v for k, v in gpu_telemetry().items() if k in ("act_freq_mhz", "temp_max_c")}
            self.log_f.write(json.dumps(rec) + "\n"); self.log_f.flush()
        return rec

    # ------------------------------------------------------------------ deterministic held-out eval
    @torch.no_grad()
    def evaluate(self, n_seqs: int | None = None) -> dict:
        c = self.cfg
        self.ts.eval()
        n = n_seqs or c.eval_seqs
        g = torch.Generator(device="xpu")
        tot, cnt, ces = 0.0, 0, []
        for i in range(n):
            g.manual_seed(10_000 + i)  # same t and masks for sequence i every time
            loss, info = self.ts.training_loss(self.seq(self.held, i), c.eval_ctx, c.block_size, weight_by_t=c.weight_by_t, generator=g)
            tot += loss.item(); cnt += 1; ces.append(info["ce_masked_mean"])
        res = {"step": self.step, "eval_loss": tot / cnt, "eval_ce_masked": float(np.mean(ces)), "n_seqs": cnt, "time": time.time()}
        self.eval_history.append(res["eval_loss"])
        self.log_f.write(json.dumps({"eval": res}) + "\n"); self.log_f.flush()
        self.ts.train()
        return res

    def should_early_stop(self) -> bool:
        p = self.cfg.early_stop_patience
        h = self.eval_history
        if not p or len(h) < p + 1:
            return False
        return all(h[-k] > h[-k - 1] for k in range(1, p + 1))

    # ------------------------------------------------------------------ checkpoints (trainable state only)
    def save(self, name: str) -> Path:
        lora = {n: p.detach().cpu() for n, p in self.ts.dream.named_parameters() if "lora_" in n and ".bridge." not in n}
        state = {"step": self.step, "tokens_seen": self.tokens_seen, "config": asdict(self.cfg),
                 "bridge": {k: v.cpu() for k, v in bridge_state_dict(self.ts.dream).items()}, "lora": lora,
                 "optimizer": self.opt.state_dict(), "eval_history": self.eval_history, "best_eval": self.best_eval}
        p = self.run_dir / f"{name}.pt"
        torch.save(state, p)
        return p

    def load_trainable(self, path: Path, load_optimizer: bool = True):
        st = torch.load(path, map_location="cpu", weights_only=False)
        layers = self.ts.dream.model.layers
        for k, v in st["bridge"].items():
            _, i, _, *rest = k.split(".")
            layers[int(i)].bridge.state_dict()[".".join(rest)].copy_(v.to("xpu"))
        named = dict(self.ts.dream.named_parameters())
        for k, v in st["lora"].items():
            named[k].data.copy_(v.to("xpu"))
        if load_optimizer and "optimizer" in st:
            self.opt.load_state_dict(st["optimizer"])
        self.step, self.tokens_seen = st["step"], st["tokens_seen"]
        self.eval_history, self.best_eval = st.get("eval_history", []), st.get("best_eval", math.inf)
        return st
