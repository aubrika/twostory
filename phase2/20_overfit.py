"""Phase 2 / 20: overfit ONE fixed batch (one sequence, fixed ctx_len, fixed mask pattern) with the real config
(full-size bridge, LoRA r=16, grad checkpointing, AdamW). Gates: loss -> ~0; peak VRAM incl. optimizer <= 28 GiB."""
import argparse, json, torch
from twostory.util import gate, finish, write_report, MemTracker, gib, provenance
from twostory.model import TwoStory, LoraSpec
from twostory.bridge import BridgeConfig
from twostory.train import Trainer, TrainConfig

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--optimizer", default="adamw")
ap.add_argument("--out-rank", type=int, default=None)
ap.add_argument("--lora-rank", type=int, default=16)
ap.add_argument("--ctx-len", type=int, default=512)
ap.add_argument("--seq", type=int, default=0)
a = ap.parse_args()

with MemTracker() as mt:
    ts = TwoStory.load(BridgeConfig(out_rank=a.out_rank), LoraSpec(rank=a.lora_rank))
print("load:", mt.summary(), ts.param_report())
cfg = TrainConfig(steps=a.steps, lr=a.lr, warmup=10, optimizer=a.optimizer, run_name=f"phase2_overfit_{a.optimizer}_r{a.out_rank}",
                  eval_every=0, log_every=1)
tr = Trainer(ts, cfg)
torch.xpu.reset_peak_memory_stats()
hist = []
for i in range(a.steps):
    rec = tr.train_step(seq_idx=[a.seq], ctx_len=a.ctx_len, reseed=1234)  # same sequence, same masks, every step
    hist.append(rec["loss"])
    if i < 5 or (i + 1) % 10 == 0:
        print(f"step {rec['step']:4d} loss {rec['loss']:.4f} lr {rec['lr']:.2e} gn {rec['grad_norm']:.2f} {rec['seconds']:.2f}s "
              f"vram {rec['vram_alloc_gib']}/{rec['vram_peak_gib']} GiB", flush=True)
peak = gib(torch.xpu.max_memory_allocated()); reserved = gib(torch.xpu.max_memory_reserved())
opt_bytes = sum(v.numel() * v.element_size() for st in tr.opt.state.values() for v in st.values() if torch.is_tensor(v))
print(f"optimizer state {gib(opt_bytes)} GiB; peak alloc {peak} GiB, peak reserved {reserved} GiB")
final = min(hist[-10:])
gate("loss drives toward ~0 (min of last 10 steps < 0.15)", final < 0.15, f"{final:.4f} (start {hist[0]:.3f})")
gate("loss fell by >= 90%", final < 0.1 * hist[0], f"{final/hist[0]:.3f} of initial")
gate("peak VRAM incl. optimizer <= 28 GiB (alloc)", peak <= 28, f"{peak} GiB alloc / {reserved} GiB reserved")
gate("no NaN/inf in loss history", all(x == x and abs(x) < 1e9 for x in hist))
write_report(f"20_overfit_{a.optimizer}_r{a.out_rank}", {"args": vars(a), "provenance": provenance(), "params": ts.param_report(),
             "loss": hist, "peak_alloc_gib": peak, "peak_reserved_gib": reserved, "optimizer_state_gib": gib(opt_bytes),
             "sec_per_step": sum(1 for _ in hist) and None, "log": str(tr.log_path)}, subdir="phase2")
finish("Phase 2 / 20 overfit one batch")
