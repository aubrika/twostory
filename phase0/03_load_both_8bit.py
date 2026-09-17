"""Phase 0 / 03: Qwen2.5-7B + Dream-7B both resident 8-bit on XPU. Gate: <= ~16 GiB VRAM, no swap."""
import argparse, torch
from twostory.util import gate, finish, write_report, MemTracker, load_qwen, load_dream

ap = argparse.ArgumentParser(); ap.add_argument("--bits", type=int, default=8); a = ap.parse_args()
with MemTracker() as m1:
    qwen = load_qwen(a.bits)
print("qwen:", m1.summary())
with MemTracker() as m2:
    dream = load_dream(a.bits)
print("dream:", m2.summary())
total = torch.xpu.memory_allocated()
print(f"both resident: {total/2**30:.2f} GiB allocated, {torch.xpu.memory_reserved()/2**30:.2f} GiB reserved")
with torch.no_grad():
    ids = torch.randint(0, 150000, (1, 64), device="xpu")
    lq = qwen(ids).logits; ld = dream(ids).logits
gate("dream loads without swap growth", m2.summary()["swap_growth_gib"] < 0.5, f"rss peak {m2.summary()['rss_peak_gib']} GiB")
gate("both towers resident <= 17 GiB", total / 2**30 <= 17, f"{total/2**30:.2f} GiB")
gate("dream forward finite on XPU", torch.isfinite(ld.float()).all().item())
gate("dream logits shape", tuple(ld.shape) == (1, 64, dream.config.vocab_size), str(tuple(ld.shape)))
write_report(f"03_load_both_{a.bits}bit", {"bits": a.bits, "qwen": m1.summary(), "dream": m2.summary(), "both_alloc_gib": round(total/2**30, 2)})
finish("Phase 0 / 03 load both")
