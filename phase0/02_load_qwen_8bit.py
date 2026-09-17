"""Phase 0 / 02: load Qwen2.5-7B 8-bit on XPU. Gate: no swap, VRAM <= ~8 GiB."""
import argparse, torch
from twostory.util import gate, finish, write_report, MemTracker, load_qwen, gib

ap = argparse.ArgumentParser(); ap.add_argument("--bits", type=int, default=8); a = ap.parse_args()
with MemTracker() as mt:
    m = load_qwen(a.bits)
s = mt.summary(); print(s)
w = m.model.layers[0].self_attn.q_proj.weight
print("q_proj weight:", w.dtype, tuple(w.shape), w.device)
with torch.no_grad(), MemTracker() as fw:
    ids = torch.randint(0, 150000, (1, 64), device="xpu")
    out = m(ids).logits
print("forward:", fw.summary())
gate("qwen loads without swap growth", s["swap_growth_gib"] < 0.5, f"swap +{s['swap_growth_gib']} GiB, rss peak {s['rss_peak_gib']} GiB")
gate("qwen resident VRAM <= 9 GiB", s["vram_now_gib"] <= 9, f"{s['vram_now_gib']} GiB")
gate("qwen weights are int8", w.dtype == torch.int8 if a.bits == 8 else True, str(w.dtype))
gate("qwen forward finite on XPU", torch.isfinite(out.float()).all().item())
write_report(f"02_load_qwen_{a.bits}bit", {"bits": a.bits, "load": s, "forward": fw.summary()})
finish("Phase 0 / 02 load qwen")
