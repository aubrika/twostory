"""Aggregate Phase 5 raw JSONL into tables and figures (docs/BENCHMARK.md §6). Never re-runs the GPU.
Reads logs/phase5/raw_{main,q,swap,gsm,gsmq}.jsonl; writes docs/figs/*.png and prints markdown tables (also saved to
logs/phase5/tables.md). Every number: median with IQR over repeats, per tier."""
import json, math, statistics, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from twostory.util import REPO

RAW = REPO / "logs/phase5"; FIGS = REPO / "docs/figs"; FIGS.mkdir(parents=True, exist_ok=True)
TIERS = ["short", "medium", "long", "vlong", "hand"]; TIER_TOK = {"short": 128, "medium": 512, "long": 2048, "vlong": 4096}

def load(name):
    p = RAW / f"raw_{name}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []

rows = load("main") + load("q") + load("swap"); gsm = load("gsm") + load("gsmq")
gen = [r for r in rows if "steady_tok_s" in r]
extras = [r for r in rows if r.get("kind")]
headers = [r for r in rows if str(r.get("key", "")).startswith("header")]

def sysname(r):
    if r["key"].startswith("sweep:"): return None
    s = r["system"]
    if s == "D": return f"D S={r['S']}"
    if s in ("T", "T0", "Tbf16"): return f"{s} S={r['S']},B={r['B']}"
    return s

def med_iqr(v):
    v = sorted(v)
    if not v: return (math.nan, math.nan)
    if len(v) < 4: return (statistics.median(v), max(v) - min(v))
    q = statistics.quantiles(v, n=4); return (statistics.median(v), q[2] - q[0])

def fmt(m, i, d=1): return f"{m:.{d}f} ({i:.{d}f})" if not math.isnan(m) else "—"

# ------------------------------------------------------------------ main table per tier
by = defaultdict(list)
for r in gen:
    n = sysname(r)
    if n: by[(r["tier"], n)].append(r)
order = ["Q", "D S=256", "D S=64", "T0 S=8,B=32", "T S=8,B=32", "T S=32,B=32", "Tbf16 S=8,B=32"]
out = []
for tier in TIERS:
    out.append(f"\n### Tier: {tier} ({TIER_TOK.get(tier, 'mixed')} prompt tokens, 256 output tokens)\n")
    out.append("| System | steady tok/s (IQR) | TTFT s | wall s | peak VRAM GiB | repetition | judge NLL no-rep | judge NLL all | n |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for n in order:
        rs = by.get((tier, n), [])
        if not rs: continue
        j = [r["judge_nll"]["norep"] for r in rs if isinstance(r["judge_nll"], dict) and not math.isnan(r["judge_nll"]["norep"])]
        ja = [r["judge_nll"]["all"] if isinstance(r["judge_nll"], dict) else r["judge_nll"] for r in rs]
        out.append(f"| {n} | {fmt(*med_iqr([r['steady_tok_s'] for r in rs]))} | {fmt(*med_iqr([r['ttft_s'] for r in rs]), 2)} | {fmt(*med_iqr([r['total_s'] for r in rs]))} | "
                   f"{statistics.median(r['peak_gib'] for r in rs):.1f} | {statistics.median(r['rep'] for r in rs):.2f} | {fmt(*med_iqr(j), 2)} | {fmt(*med_iqr(ja), 2)} | {len(rs)} |")

# ------------------------------------------------------------------ prompt-cost scaling: wall-clock vs prompt length, slope per system
out.append("\n### Prompt-cost scaling (median wall-clock per tier; slope = seconds per 1000 prompt tokens, linear fit over the four length tiers)\n")
out.append("| System | 128 | 512 | 2048 | 4096 | slope s/1k tok |"); out.append("|---|---|---|---|---|---|")
scal = {}
for n in order:
    xs, ys = [], []
    cells = []
    for tier in ["short", "medium", "long", "vlong"]:
        rs = by.get((tier, n), [])
        if rs: m = statistics.median(r["total_s"] for r in rs); xs.append(TIER_TOK[tier]); ys.append(m); cells.append(f"{m:.1f}")
        else: cells.append("—")
    if len(xs) >= 2:
        slope = float(np.polyfit(xs, ys, 1)[0]) * 1000; scal[n] = (xs, ys, slope)
        out.append(f"| {n} | " + " | ".join(cells) + f" | {slope:.2f} |")

# ------------------------------------------------------------------ held-out CE, long-context retention, Q ppl, GSM
ce = {(r["system"], r.get("B")): r["ce"] for r in extras if r["kind"] == "heldout_ce"}
ref = json.load(open(REPO / "logs/phase3/32_reference_loss.json")) if (REPO / "logs/phase3/32_reference_loss.json").exists() else {}
D_ref = min(ref.get("D_full_context_bos0", math.nan), ref.get("D_full_context_bos1", math.nan)) if ref else math.nan
out.append("\n### Quality: held-out masked-block CE (identical 32 sequences and masks; lower is better; independent of S)\n")
out.append("| System | CE | gap closed toward D |"); out.append("|---|---|---|")
if ref: out.append(f"| D (plain Dream, context in-input) | {D_ref:.3f} | 100% |")
for (s, B), v in sorted(ce.items()):
    gc = (ref["T0_untrained"] - v) / (ref["T0_untrained"] - D_ref) if ref else math.nan
    out.append(f"| {s} (eval block size {B}) | {v:.3f} | {100*gc:.1f}% |")
q_ppl = [r for r in extras if r["kind"] == "heldout_ppl"]
if q_ppl: out.append(f"\nQ held-out AR perplexity (separate metric, not comparable to masked CE): **{q_ppl[0]['ppl']:.2f}** (NLL/token {q_ppl[0]['nll_per_token']:.3f}, {q_ppl[0]['n_tokens']} tokens)")
lc = defaultdict(dict)
for r in extras:
    if r["kind"] == "longctx_ce": lc[r["ctx_len"]][r["system"]] = r["ce"]
if lc:
    out.append("\n### Long-context quality retention: masked CE of 4 blocks after N context tokens (16 held-out passages)\n")
    out.append("| context tokens | D (context in-input) | T (context via bridge) | gap |"); out.append("|---|---|---|---|")
    for c in sorted(lc): out.append(f"| {c} | {lc[c].get('D', math.nan):.3f} | {lc[c].get('T', math.nan):.3f} | {lc[c].get('T', math.nan) - lc[c].get('D', math.nan):+.3f} |")
if gsm:
    out.append("\n### GSM8K sanity read (200 items, 4-shot, exact match on the final number; secondary metric)\n")
    out.append("| System | accuracy | n |"); out.append("|---|---|---|")
    g = defaultdict(list)
    for r in gsm:
        if r.get("kind") == "gsm": g[r["system"] + (f" S={r['S']}" if r.get("S") else "")].append(bool(r["correct"]))
    for k, v in g.items(): out.append(f"| {k} | {100*np.mean(v):.1f}% | {len(v)} |")

# ------------------------------------------------------------------ sweep -> Pareto (Long tier)
sw = [r for r in rows if str(r.get("key", "")).startswith("sweep:")]
pts = defaultdict(list)
for r in sw: pts[(r["system"], r["S"], r["B"])].append(r)
pareto = []
for (s, S, B), rs in sorted(pts.items()):
    j = [r["judge_nll"]["norep"] for r in rs if isinstance(r["judge_nll"], dict) and not math.isnan(r["judge_nll"]["norep"])]
    pareto.append({"system": s, "S": S, "B": B, "tok_s": statistics.median(r["steady_tok_s"] for r in rs), "rep": statistics.median(r["rep"] for r in rs),
                   "judge_norep": statistics.median(j) if j else math.nan, "n": len(rs)})
if pareto:
    out.append("\n### Operating-point sweep at the Long tier (2048-token prompts, 256 output tokens; medians)\n")
    out.append("| System | S | B | steady tok/s | repetition | judge NLL no-rep |"); out.append("|---|---|---|---|---|---|")
    for p in pareto: out.append(f"| {p['system']} | {p['S']} | {p['B'] or '—'} | {p['tok_s']:.1f} | {p['rep']:.2f} | {p['judge_norep']:.2f} |")

# ------------------------------------------------------------------ figures
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if pareto:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, yk, yl in ((axes[0], "rep", "repetition rate (lower is better)"), (axes[1], "judge_norep", "judge NLL on non-repeated tokens (lower is better)")):
            for s, mk, col in (("T", "o", "#1f77b4"), ("D", "s", "#d62728")):
                P = [p for p in pareto if p["system"] == s]
                ax.scatter([p["tok_s"] for p in P], [p[yk] for p in P], marker=mk, color=col, label=f"{s} sweep")
                for p in P: ax.annotate(f"S{p['S']}" + (f"B{p['B']}" if p["B"] else ""), (p["tok_s"], p[yk]), fontsize=7, alpha=0.8)
            qrs = by.get(("long", "Q"), [])
            if qrs:
                jq = [r["judge_nll"]["norep"] for r in qrs if isinstance(r["judge_nll"], dict)]
                ax.scatter([statistics.median(r["steady_tok_s"] for r in qrs)], [statistics.median(r["rep"] for r in qrs) if yk == "rep" else statistics.median(jq)], marker="*", s=160, color="#2ca02c", label="Q (AR, greedy)")
            ax.set_xlabel("steady-state tok/s (2048-token prompts)"); ax.set_ylabel(yl); ax.grid(alpha=0.3); ax.legend()
        fig.suptitle("TwoStory vs plain Dream: quality vs throughput at the Long tier"); fig.tight_layout(); fig.savefig(FIGS / "pareto_long.png", dpi=150)
    if scal:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for n, (xs, ys, slope) in scal.items(): ax.plot(xs, ys, marker="o", label=f"{n} ({slope:.2f} s/1k tok)")
        ax.set_xlabel("prompt tokens"); ax.set_ylabel("median wall-clock for 256 output tokens (s)"); ax.set_xscale("log", base=2); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title("Prompt-cost scaling"); fig.tight_layout(); fig.savefig(FIGS / "scaling.png", dpi=150)
    if lc:
        fig, ax = plt.subplots(figsize=(6, 4))
        cs = sorted(lc); ax.plot(cs, [lc[c].get("D") for c in cs], marker="s", color="#d62728", label="D: context in-input"); ax.plot(cs, [lc[c].get("T") for c in cs], marker="o", color="#1f77b4", label="T: context via bridge")
        ax.set_xscale("log", base=2); ax.set_xlabel("context tokens"); ax.set_ylabel("masked CE of the next 4 blocks"); ax.grid(alpha=0.3); ax.legend(); ax.set_title("Long-context retention"); fig.tight_layout(); fig.savefig(FIGS / "longctx.png", dpi=150)
except Exception as e:
    out.append(f"\n(figure error: {e})")

prov = headers[0]["provenance"] if headers else {}
out.insert(0, f"Provenance: commit {prov.get('commit')} (dirty={prov.get('dirty')}), {prov.get('date')}, torch {prov.get('versions', {}).get('torch')}, prompt set sha256 {headers[0]['prompt_set_sha256'][:16] if headers else '?'}…, other GPU users at start: {headers[0].get('gpu_other_users') if headers else '?'}")
text = "\n".join(out); (RAW / "tables.md").write_text(text); print(text)
