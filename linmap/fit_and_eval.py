"""Linear K/V-map bridge (2026-09-20). No gradient training anywhere.
Stage A: capture pre-RoPE K and V per layer from both towers on the same text; closed-form ridge regression per layer
         Qwen K/V (512-d) -> Dream K/V (512-d), and Qwen attention-input hidden (3584-d) -> Dream K/V; report held-out R^2 by layer
         (identity / linear-KV / linear-hidden). This IS the measurement of the protocol's lineage premise.
Stage B: feed mapped K/V into Dream's OWN attention (concat, NVIDIA-TwoTower style) and score the standard 32 held-out sequences
         with the validation masks used since Phase 3, plus argmax agreement with Dream.
         The trained cross-attention checkpoint is not evaluated by this script.
Frames: Dream sees [BOS, ctx..., block]; context token j sits at Dream position j+1 and Qwen position j. In Stage B the denoiser input is
[BOS, anchor=last ctx token, block]; external K/V cover ctx tokens 0..s-2, RoPE'd at Dream positions 1..s-1."""
import argparse, json, math, time
import numpy as np, torch, torch.nn.functional as F
from twostory.util import REPO, load_qwen, gib, provenance
from twostory.model import load_dream_vendored
from twostory.dream import rotate_half

ap = argparse.ArgumentParser()
ap.add_argument("--fit-seqs", type=int, default=160); ap.add_argument("--val-seqs", type=int, default=24); ap.add_argument("--eval-seqs", type=int, default=32)
ap.add_argument("--ridge", type=float, default=1e-3); ap.add_argument("--no-hidden", action="store_true"); ap.add_argument("--block", type=int, default=32)
a = ap.parse_args()
if a.fit_seqs < 1 or a.val_seqs < 1 or not 1 <= a.eval_seqs <= 100:
    ap.error("fit/val counts must be positive; eval-seqs must be 1..100 to avoid the R2 validation slice")
if a.block < 1 or 512 % a.block:
    ap.error("block must be a positive divisor of the 512-token evaluation suffix")
for required in ("train.bin", "heldout.bin", "meta.json"):
    if not (REPO / "data/tokens" / required).is_file():
        ap.error(f"Missing data/tokens/{required}; run phase0/07_pretokenize.py first")
dev, BOS, B = "xpu", 151665, a.block
OUT = REPO / "logs/linmap"; OUT.mkdir(parents=True, exist_ok=True)
t00 = time.time()
dream = load_dream_vendored(16); qwen = load_qwen(8); MASK = dream.config.mask_token_id; NL = dream.config.num_hidden_layers
print(f"loaded: {gib(torch.xpu.memory_allocated())} GiB", flush=True)
meta = json.load(open(REPO / "data/tokens/meta.json")); L = meta["seq_len"]
train = np.memmap(REPO / "data/tokens/train.bin", dtype=np.uint32, mode="r").reshape(-1, L)
held = np.memmap(REPO / "data/tokens/heldout.bin", dtype=np.uint32, mode="r").reshape(-1, L)

# ------------------------------------------------------------------ capture hooks (pre-RoPE k_proj / v_proj outputs; Qwen attention input)
capD, capQ = {}, {}
for i, l in enumerate(dream.model.layers):
    l.self_attn.k_proj.register_forward_hook(lambda m, inp, out, i=i: capD.__setitem__(("k", i), out.detach()))
    l.self_attn.v_proj.register_forward_hook(lambda m, inp, out, i=i: capD.__setitem__(("v", i), out.detach()))
for i, l in enumerate(qwen.model.layers):
    l.self_attn.k_proj.register_forward_hook(lambda m, inp, out, i=i: capQ.__setitem__(("k", i), out.detach()))
    l.self_attn.v_proj.register_forward_hook(lambda m, inp, out, i=i: capQ.__setitem__(("v", i), out.detach()))
    l.self_attn.k_proj.register_forward_pre_hook(lambda m, inp, i=i: capQ.__setitem__(("h", i), inp[0].detach()))

@torch.no_grad()
def run_dream_ctx(ctx):
    """Dream over [BOS] + ctx + fully masked block (the deployable condition at block start). Returns per-layer (K_pre, V_pre) at ctx positions."""
    x = torch.tensor([[BOS] + ctx + [MASK] * B], device=dev); dream(x)
    c = len(ctx); return [(capD[("k", i)][0, 1:1 + c].float(), capD[("v", i)][0, 1:1 + c].float()) for i in range(NL)]

@torch.no_grad()
def run_qwen(ctx):
    x = torch.tensor([ctx], device=dev); qwen.model(input_ids=x, use_cache=False)
    return [(capQ[("k", i)][0].float(), capQ[("v", i)][0].float(), capQ[("h", i)][0].float()) for i in range(NL)]

# ------------------------------------------------------------------ Stage A: fit
def aug(X): return torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1)
acc = {(src, i): None for src in ("kv_k", "kv_v") + (() if a.no_hidden else ("h",)) for i in range(NL)}
gpuA = {}; gpuB = {}
def add(key, X, Y):
    Xa = aug(X); A_, B_ = Xa.T @ Xa, Xa.T @ Y
    if key in gpuA: gpuA[key] += A_; gpuB[key] += B_
    else: gpuA[key], gpuB[key] = A_, B_
cpuA, cpuB = {}, {}
def flush():
    # Only the small 513x513 K/V accumulators go to CPU fp64. The 3585x3585 hidden-state accumulators stay on the GPU in fp32:
    # 28 of them in fp64 on the CPU (plus per-flush temporaries) blew through a 12 GB cap on this 15 GB box (2026-09-20).
    for k in [k for k in gpuA if k[0] != "h"]:
        A_, B_ = gpuA.pop(k).double().cpu(), gpuB.pop(k).double().cpu()
        if k in cpuA: cpuA[k] += A_; cpuB[k] += B_
        else: cpuA[k], cpuB[k] = A_, B_
rng = np.random.default_rng(0); n_tok = 0; t0 = time.time()
for n in range(a.fit_seqs):
    seq = train[int(rng.integers(0, len(train)))].astype(np.int64); c = int(rng.integers(8, 25)) * 32
    ctx = seq[:c].tolist(); D = run_dream_ctx(ctx); Q = run_qwen(ctx); n_tok += c
    for i in range(NL):
        add(("kv_k", i), Q[i][0], D[i][0]); add(("kv_v", i), Q[i][1], D[i][1])
        if not a.no_hidden: add(("h", i), Q[i][2], torch.cat([D[i][0], D[i][1]], 1))   # hidden -> [K;V] jointly (shared X^T X)
    if (n + 1) % 8 == 0: flush()
    if (n + 1) % 40 == 0: print(f"  fit {n+1}/{a.fit_seqs} seqs, {n_tok} tokens, {time.time()-t0:.0f}s", flush=True)
flush()
W = {}
for k in list(cpuA) + [k for k in list(gpuA) if k[0] == "h"]:                              # one map at a time; free as we go
    if k in cpuA: A_, B_ = cpuA.pop(k), cpuB.pop(k)
    else: A_, B_ = gpuA.pop(k).double().cpu(), gpuB.pop(k).double().cpu()
    lam = a.ridge * torch.trace(A_) / A_.shape[0]
    A_.diagonal()[:-1].add_(lam)                                                          # ridge in place (bias unregularised)
    W[k] = torch.linalg.solve(A_, B_).float().to(dev); del A_, B_                         # [din+1, dout]
print(f"Stage A fit done: {n_tok} tokens, {len(W)} maps, {time.time()-t0:.0f}s", flush=True)
def map_kv(kind, i, Qi):
    """kind in identity|linear_kv|linear_h -> (K_pre, V_pre) in Dream space, float32 [T, 512]"""
    if kind == "identity": return Qi[0], Qi[1]
    if kind == "linear_kv": return aug(Qi[0]) @ W[("kv_k", i)], aug(Qi[1]) @ W[("kv_v", i)]
    Y = aug(Qi[2]) @ W[("h", i)]; return Y[:, :512], Y[:, 512:]
kinds = ["identity", "linear_kv"] + ([] if a.no_hidden else ["linear_h"])
# R2 uses indices 100 onward. Indices 0..31 are reused training-validation data, not a final test set.
# STREAMING: holding 24 sequences of captures on the GPU (7.2 GiB incl. Qwen's 3584-d hidden states) overflowed VRAM, the xe driver
# spilled into system RAM, and the job was OOM-killed twice (2026-09-20). Accumulate sums only.
def memlog(tag):
    free = torch.xpu.mem_get_info()[0] / 2**30; print(f"[mem] {tag}: torch alloc {gib(torch.xpu.memory_allocated())} GiB, device free {free:.2f} GiB", flush=True)
    assert free > 0.8, f"VRAM guard: only {free:.2f} GiB free at {tag}"
memlog("after fit")
sse = {(kd, kv, i): 0.0 for kd in kinds for kv in "kv" for i in range(NL)}; cosv = {(kd, kv, i): 0.0 for kd in kinds for kv in "kv" for i in range(NL)}
sumY = {(kv, i): 0.0 for kv in "kv" for i in range(NL)}; sumY2 = {(kv, i): 0.0 for kv in "kv" for i in range(NL)}; nval = 0
for n in range(a.val_seqs):
    seq = held[100 + n].astype(np.int64); ctx = seq[:512].tolist(); Dv, Qv = run_dream_ctx(ctx), run_qwen(ctx); nval += len(ctx)
    for i in range(NL):
        for kv_i, kv in enumerate("kv"):
            Y = Dv[i][kv_i]; sumY[(kv, i)] = sumY[(kv, i)] + Y.sum(0).double().cpu(); sumY2[(kv, i)] += float((Y.double() ** 2).sum())
            for kd in kinds:
                P = map_kv(kd, i, Qv[i])[kv_i]
                sse[(kd, kv, i)] += float(((Y - P).double() ** 2).sum()); cosv[(kd, kv, i)] += float(F.cosine_similarity(Y, P, dim=1).sum())
    del Dv, Qv
sst = {k: sumY2[k] - float((sumY[k] ** 2).sum()) / nval for k in sumY}
for k in cosv: cosv[k] /= nval
memlog("after validation")
r2 = {kd: {kv: [1 - sse[(kd, kv, i)] / sst[(kv, i)] for i in range(NL)] for kv in "kv"} for kd in kinds}
cos = {kd: {kv: [cosv[(kd, kv, i)] for i in range(NL)] for kv in "kv"} for kd in kinds}
print("\nStage A: held-out R^2 of Dream's context K/V predicted from Qwen (pre-RoPE), by layer")
print("layer | " + " | ".join(f"{kd}:K   {kd}:V " for kd in kinds))
for i in range(NL): print(f"  {i:2d}  | " + " | ".join(f"{r2[kd]['k'][i]:7.3f} {r2[kd]['v'][i]:7.3f}" for kd in kinds))
print(" mean | " + " | ".join(f"{np.mean(r2[kd]['k']):7.3f} {np.mean(r2[kd]['v']):7.3f}" for kd in kinds), flush=True)

# ------------------------------------------------------------------ Stage B: functional test (concat into Dream's own attention)
for l in dream.model.layers: l.kv_concat = True
def rope_k(Kpre, positions):
    k = Kpre.view(1, -1, dream.config.num_key_value_heads, 128).transpose(1, 2)                     # [1, 4, T, 128]
    cos_, sin_ = dream.model.rotary_emb(k, positions[None]); cos_, sin_ = cos_.unsqueeze(1).float(), sin_.unsqueeze(1).float()
    return k * cos_ + rotate_half(k) * sin_
def as_v(Vpre): return Vpre.view(1, -1, dream.config.num_key_value_heads, 128).transpose(1, 2)

@torch.no_grad()
def denoise_logits(anchor, blk, s, ext):
    x = torch.cat([torch.tensor([BOS, anchor], device=dev), blk])[None]
    pos = torch.cat([torch.tensor([0], device=dev), torch.arange(s, s + B + 1, device=dev)])[None]   # BOS@0, anchor@s, block@s+1..
    return dream(input_ids=x, position_ids=pos, ctx_kv=ext).logits[0, 1:1 + B].float()              # hidden at x[1+k] predicts block token k

variants = ["none"] + kinds + ["oracle_own_kv"]
tot = {v: 0.0 for v in variants}; agree = {v: 0 for v in variants}; tot["D_same_code"] = 0.0; cnt = 0; t0 = time.time()
for n in range(a.eval_seqs):
    seq = torch.from_numpy(held[n].astype(np.int64)).to(dev); seql = seq.tolist(); ctx0 = 512; nb = (L - ctx0) // B
    g = torch.Generator(device=dev); g.manual_seed(10_000 + n)                                       # identical masks to Trainer.evaluate / phase3 reference
    t = torch.rand(nb, device=dev, generator=g).clamp_(1e-3, 1.0); masked = torch.rand(nb, B, device=dev, generator=g) < t[:, None]
    none_ = ~masked.any(1); masked[none_, torch.randint(0, B, (int(none_.sum()),), device=dev, generator=g)] = True
    Qfull = run_qwen(seql)                                                                          # causal => prefix-consistent; slice per block
    for j in range(nb):
        s = ctx0 + j * B; tgt = seq[s:s + B]; blk = torch.where(masked[j], torch.full((B,), MASK, device=dev), tgt); m = masked[j]
        for l in dream.model.layers: l.kv_concat = False
        lgD = dream(torch.cat([torch.tensor([BOS], device=dev), seq[:s], blk])[None]).logits[0, s: s + B].float()   # plain Dream, true block state
        Dkv = run_dream_ctx(seql[:s])                                                               # oracle: Dream's own ctx K/V, block fully masked
        for l in dream.model.layers: l.kv_concat = True
        aD = lgD.argmax(-1); tot["D_same_code"] += float(F.cross_entropy(lgD[m], tgt[m], reduction="sum"))
        posc = torch.arange(1, s, device=dev)                                                       # ctx tokens 0..s-2 at Dream positions 1..s-1
        for v in variants:
            if v == "none": ext = None
            elif v == "oracle_own_kv": ext = [(rope_k(Dkv[i][0][: s - 1], posc), as_v(Dkv[i][1][: s - 1])) for i in range(NL)]
            else:
                ext = []
                for i in range(NL):
                    Kp, Vp = map_kv(v, i, tuple(z[: s - 1] for z in Qfull[i])); ext.append((rope_k(Kp, posc), as_v(Vp)))
            lg = denoise_logits(seql[s - 1], blk, s, ext)
            tot[v] += float(F.cross_entropy(lg[m], tgt[m], reduction="sum")); agree[v] += int((lg.argmax(-1)[m] == aD[m]).sum())
        cnt += int(m.sum())
    if n == 0: memlog("first eval sequence")
    if (n + 1) % 4 == 0: print(f"  eval {n+1}/{a.eval_seqs}: " + "  ".join(f"{v} {tot[v]/cnt:.3f}" for v in variants) + f"  D {tot['D_same_code']/cnt:.3f}  ({time.time()-t0:.0f}s)", flush=True)
res = {"provenance": provenance(), "args": vars(a), "fit_tokens": n_tok, "r2": r2, "cos": cos,
       "ce": {v: tot[v] / cnt for v in tot}, "agree_with_D": {v: agree[v] / cnt for v in variants}, "n_masked": cnt,
       "evaluation": {"split_role": "validation", "sequence_start": 0, "sequence_count": a.eval_seqs,
                      "mask_seed_start": 10000, "trained_bridge_evaluated": False},
       "minutes": round((time.time() - t00) / 60, 1)}
print("\nStage B: validation masked CE and argmax agreement with Dream")
print("The trained cross-attention checkpoint is not evaluated in this run.")
for v in variants:
    gc = (res["ce"]["none"] - res["ce"][v]) / (res["ce"]["none"] - res["ce"]["D_same_code"])
    print(f"  {v:15s} CE {res['ce'][v]:.3f}   gap closed vs 'none' {100*gc:6.1f}%   agreement with Dream {100*res['agree_with_D'][v]:.1f}%")
json.dump(res, open(OUT / f"linmap_results_fit{a.fit_seqs}.json", "w"), indent=1, default=str)
torch.save({k: v.cpu() for k, v in W.items()}, OUT / f"linmap_weights_fit{a.fit_seqs}.pt")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for kv_i, kv in enumerate("kv"):
        for kd, c in zip(kinds, ("#7f7f7f", "#1f77b4", "#2ca02c")): ax[kv_i].plot(range(NL), r2[kd][kv], marker="o", ms=3, color=c, label=kd)
        ax[kv_i].set_title(f"Dream {'keys' if kv=='k' else 'values'} predicted from Qwen (held-out R²)"); ax[kv_i].set_xlabel("layer"); ax[kv_i].grid(alpha=.3); ax[kv_i].set_ylim(-0.5, 1.02)
    ax[0].set_ylabel("R²"); ax[0].legend(); fig.tight_layout(); (REPO / "docs/figs").mkdir(exist_ok=True); fig.savefig(REPO / "docs/figs/linmap_r2.png", dpi=150)
except Exception as e: print("figure error", e)
print("LINMAP COMPLETE", flush=True)
