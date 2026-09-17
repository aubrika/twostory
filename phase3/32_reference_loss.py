"""Phase 3 / 32: what did the bridge cost? Held-out masked-block CE on IDENTICAL blocks and masks for
  D   = plain Dream with the context in its own input (bidirectional over context + block)  -> the ceiling T chases
  T   = TwoStory best checkpoint (context only via the bridge)
  T0  = TwoStory untrained (no context at all)                                              -> the floor
plus bridged generation samples at S in {8,16,32} steps per 32-token block (repetition is mostly a steps-per-token effect)."""
import argparse, json, numpy as np, torch, torch.nn.functional as F
from twostory.util import write_report, load_tokenizer, REPO, provenance
from twostory.model import TwoStory, LoraSpec
from twostory.bridge import BridgeConfig
from twostory.train import Trainer, TrainConfig
from twostory.prompts import PROMPT_SHORT
from peft.tuners.tuners_utils import BaseTunerLayer

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="checkpoints/phase3_pilot/best.pt")
ap.add_argument("--n-seqs", type=int, default=32); ap.add_argument("--ctx", type=int, default=512); a = ap.parse_args()
ts = TwoStory.load(BridgeConfig(), LoraSpec(rank=16)); tr = Trainer(ts, TrainConfig(run_name="ref_tmp", grad_checkpointing=False))
ts.eval(); B, L, dev = 32, tr.L, "xpu"
def set_lora(on):
    for m in ts.dream.modules():
        if isinstance(m, BaseTunerLayer): m.enable_adapters(on)

def masks_for(i, n):  # identical RNG order to TwoStory.training_loss under Trainer.evaluate
    g = torch.Generator(device=dev); g.manual_seed(10_000 + i)
    t = torch.rand(n, device=dev, generator=g).clamp_(1e-3, 1.0)
    masked = torch.rand(n, B, device=dev, generator=g) < t[:, None]
    none = ~masked.any(1); masked[none, torch.randint(0, B, (int(none.sum()),), device=dev, generator=g)] = True
    return masked

@torch.no_grad()
def ce_T(i):
    g = torch.Generator(device=dev); g.manual_seed(10_000 + i)
    _, info = ts.training_loss(tr.seq(tr.held, i), a.ctx, B, generator=g)
    return info["ce_masked_mean"], info["n_masked"]

@torch.no_grad()
def ce_D(i, bos):
    seq = tr.seq(tr.held, i)[0]; n = (L - a.ctx) // B; masked = masks_for(i, n); tot, cnt = 0.0, 0
    for j in range(n):
        s = a.ctx + j * B
        blk = torch.where(masked[j], torch.full((B,), ts.mask_id, device=dev), seq[s:s + B])
        x = torch.cat([seq[:s], blk])
        if bos: x = torch.cat([torch.tensor([151665], device=dev), x])
        logits = ts.dream(x[None]).logits[0]
        off = 1 if bos else 0
        pred = logits[s - 1 + off: s + B - 1 + off].float()              # hidden p-1 predicts token p
        ce = F.cross_entropy(pred, seq[s:s + B], reduction="none")
        tot += ce[masked[j]].sum().item(); cnt += int(masked[j].sum())
    return tot / cnt, cnt

res = {}
set_lora(False); ts.dream._ambient_ctx = (None, None)
for bos in (False, True):
    v = [ce_D(i, bos) for i in range(a.n_seqs)]
    res[f"D_full_context_bos{int(bos)}"] = sum(c * k for c, k in v) / sum(k for _, k in v)
    print(f"D (context visible, bos={bos}): masked CE {res[f'D_full_context_bos{int(bos)}']:.4f}", flush=True)
v = [ce_T(i) for i in range(a.n_seqs)]; res["T0_untrained"] = sum(c * k for c, k in v) / sum(k for _, k in v)
print(f"T0 (untrained bridge, no context): masked CE {res['T0_untrained']:.4f}", flush=True)
tr.load_trainable(a.ckpt, load_optimizer=False); set_lora(True)
v = [ce_T(i) for i in range(a.n_seqs)]; res["T_pilot_best"] = sum(c * k for c, k in v) / sum(k for _, k in v)
print(f"T  (pilot best ckpt): masked CE {res['T_pilot_best']:.4f}", flush=True)
D = min(res["D_full_context_bos0"], res["D_full_context_bos1"])
res["gap_closed_fraction"] = (res["T0_untrained"] - res["T_pilot_best"]) / (res["T0_untrained"] - D)
print(f"gap closed: {100*res['gap_closed_fraction']:.1f}% of the way from T0 to D after 8M tokens", flush=True)

# steps-per-block sweep on the fixed sample prompts
tokq = load_tokenizer("qwen"); pset = json.load(open(REPO / "data/prompt_set.json"))["prompts"]
prompts = [("protocol_short", tokq(PROMPT_SHORT).input_ids)] + [(p["id"], p["ids"]) for p in pset if p["id"] in ("short_00", "short_07", "medium_03", "hand_story_open")]
def rep(ids): return sum(1 for k in range(2, len(ids)) if ids[k] == ids[k-1] or ids[k] == ids[k-2]) / max(1, len(ids) - 2)
ts.bridges_to(torch.bfloat16); res["steps_sweep"] = {}
for S in (8, 16, 32):
    outs = []
    for name, ids in prompts:
        r = ts.generate_block(torch.tensor([ids], device=dev), max_new_tokens=64, block_size=32, steps=S)
        outs.append({"prompt": name, "rep": round(rep(r["ids"]), 3), "tok_s": round(len(r["ids"]) / r["total_seconds"], 1), "text": tokq.decode(r["ids"])})
    res["steps_sweep"][S] = outs
    print(f"\nS={S}: mean rep {np.mean([o['rep'] for o in outs]):.3f}, mean tok/s {np.mean([o['tok_s'] for o in outs]):.1f}")
    for o in outs[:2] + outs[-1:]: print(f"   [{o['prompt']}] {o['text'][:170]!r}")
res["provenance"] = provenance(); write_report("32_reference_loss", res, subdir="phase3"); print("REFERENCE COMPLETE")
