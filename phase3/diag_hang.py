"""Diagnose the pilot's step-2 GPU job timeout and the step-0 eval NaN. Reproduces the exact draws (seed 0)."""
import sys, time, json, math, numpy as np, torch
from twostory.util import load_tokenizer, REPO, gib
from twostory.model import TwoStory, LoraSpec
from twostory.bridge import BridgeConfig
from twostory.train import Trainer, TrainConfig

MIN_FREE = [1e9]
def stage(msg):
    torch.xpu.synchronize(); free, total = torch.xpu.mem_get_info(); MIN_FREE[0] = min(MIN_FREE[0], free / 2**30)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}   | dev free {free/2**30:.2f} GiB, torch alloc {gib(torch.xpu.memory_allocated())} peak {gib(torch.xpu.max_memory_allocated())} reserved-peak {gib(torch.xpu.max_memory_reserved())}", flush=True)

ts = TwoStory.load(BridgeConfig(), LoraSpec(rank=16)); stage("loaded")
cfg = TrainConfig(steps=4000, batch_seqs=2, lr=1e-4, warmup=200, eval_seqs=32, run_name="diag_hang", log_every=1)
tr = Trainer(ts, cfg)
mode = sys.argv[1] if len(sys.argv) > 1 else "all"

if mode in ("all", "eval"):
    ts.eval(); ts.dream.gradient_checkpointing_disable()
    g = torch.Generator(device="xpu")
    bad = []
    with torch.no_grad():
        for i in range(32):
            g.manual_seed(10_000 + i)
            t0 = time.perf_counter()
            loss, info = ts.training_loss(tr.seq(tr.held, i), 512, 32, generator=g)
            torch.xpu.synchronize()
            ok = math.isfinite(loss.item())
            print(f"  eval seq {i:2d}: loss {loss.item():.4f} ce {info['ce_masked_mean']:.4f} masked {info['n_masked']} {time.perf_counter()-t0:.2f}s {'OK' if ok else 'NaN!'}", flush=True)
            if not ok: bad.append(i)
    stage(f"eval done, NaN sequences: {bad}")
    if bad:
        i = bad[0]; seq = tr.seq(tr.held, i)
        toks = seq[0].tolist(); print("  first NaN seq head:", toks[:20], "unique:", len(set(toks)), "max id", max(toks))
        with torch.no_grad():
            ts.ctx.build(seq); kv = ts.ctx.kv()
            for li, (k, v) in enumerate(kv):
                if not (torch.isfinite(k).all() and torch.isfinite(v).all()):
                    print(f"  context tower layer {li} K/V non-finite!"); break
            else:
                print("  context tower cache all finite")
            x = torch.tensor([toks[511:544]], device="xpu"); pos = torch.arange(511, 544, device="xpu").unsqueeze(0)
            out = ts.dream(x, position_ids=pos).logits; print("  dream plain forward on block finite:", torch.isfinite(out).all().item())
            out = ts.dream(x, position_ids=pos, ctx_kv=kv).logits; print("  dream + bridge (zero) forward finite:", torch.isfinite(out).all().item())
            g.manual_seed(10_000 + i)
            loss, info = ts.training_loss(seq, 512, 32, generator=g)
            print("  training_loss again:", loss.item())
    ts.dream.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); ts.train()

if mode in ("all", "steps", "stepsworst"):
    ts.train()  # gradient checkpointing only engages in training mode
    # replay the pilot's draws: Trainer.rng seeded 0; step k draws batch_seqs indices then a ctx_len per sequence
    for step in range(1, 4):
        seq_idx = tr.rng.integers(0, len(tr.train), size=cfg.batch_seqs).tolist()
        stage(f"step {step}: seq_idx {seq_idx}")
        tr.opt.zero_grad(set_to_none=True)
        for i in seq_idx:
            cl = tr.sample_ctx_len(); seq = tr.seq(tr.train, i)
            stage(f"  seq {i} ctx_len {cl} -> {(1024-cl)//32} blocks; building cache")
            with torch.no_grad():
                ts.ctx.build(seq); kv = ts.ctx.kv()
            stage("  cache built; forward+loss")
            loss, info = ts.training_loss(seq, cl, 32, generator=tr.gen)
            stage(f"  loss {loss.item():.4f} ({info}); backward")
            (loss / 2).backward()
            stage("  backward done")
        gn = torch.nn.utils.clip_grad_norm_(tr.params, 1.0).item(); stage(f"  clipped, grad norm {gn:.3f}; optimizer step")
        tr.opt.step(); stage(f"  optimizer done; vram {gib(torch.xpu.memory_allocated())}/{gib(torch.xpu.max_memory_allocated())} GiB")
if mode in ("worst", "stepsworst"):
    stage("WORST CASE: ctx_len 256 -> 24 blocks, 2 seqs/step, 6 steps")
    for step in range(6):
        t0 = time.perf_counter()
        rec = tr.train_step(ctx_len=256)
        stage(f"  worst step {step+1}: loss {rec['loss']:.4f} {time.perf_counter()-t0:.2f}s")
    tr.ts.eval(); tr.ts.dream.gradient_checkpointing_disable()
    ev = tr.evaluate(8); stage(f"eval(8 seqs) {ev['eval_loss']:.4f}")
print(f"MIN DEVICE FREE {MIN_FREE[0]:.2f} GiB")
print("DIAG COMPLETE")
