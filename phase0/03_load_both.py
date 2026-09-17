"""Phase 0 / 03: both towers resident on XPU in the training memory plan (context int8, denoiser bf16).
Gate: <= 24 GiB VRAM, no swap. (Original all-int8 plan measured 16.24 GiB; see docs/PHASE0_REPORT.md.)"""
import argparse, torch
from twostory.util import gate, finish, write_report, MemTracker, load_qwen, load_dream, CONTEXT_BITS, DENOISER_BITS

ap = argparse.ArgumentParser(); ap.add_argument("--context-bits", type=int, default=CONTEXT_BITS); ap.add_argument("--denoiser-bits", type=int, default=DENOISER_BITS); a = ap.parse_args()
with MemTracker() as m1:
    qwen = load_qwen(a.context_bits)
print("qwen:", m1.summary())
with MemTracker() as m2:
    dream = load_dream(a.denoiser_bits)
print("dream:", m2.summary())
total = torch.xpu.memory_allocated()
print(f"both resident: {total/2**30:.2f} GiB allocated, {torch.xpu.memory_reserved()/2**30:.2f} GiB reserved")
with torch.no_grad():
    ids = torch.randint(0, 150000, (1, 64), device="xpu")
    lq = qwen(ids).logits; ld = dream(ids).logits
gate("dream loads without swap growth", m2.summary()["swap_growth_gib"] < 0.5, f"rss peak {m2.summary()['rss_peak_gib']} GiB")
gate("both towers resident <= 24 GiB", total / 2**30 <= 24, f"{total/2**30:.2f} GiB")
gate("dream forward finite on XPU", torch.isfinite(ld.float()).all().item())
gate("dream logits shape", tuple(ld.shape) == (1, 64, dream.config.vocab_size), str(tuple(ld.shape)))
write_report(f"03_load_both_ctx{a.context_bits}_den{a.denoiser_bits}", {"context_bits": a.context_bits, "denoiser_bits": a.denoiser_bits, "qwen": m1.summary(), "dream": m2.summary(), "both_alloc_gib": round(total/2**30, 2)})
finish("Phase 0 / 03 load both")
