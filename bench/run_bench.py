"""Phase 5 runner. One process = one session (models resident once). Interleaves systems per prompt so thermal drift hits all
systems equally; warm-up per system; N repeats; raw JSONL with telemetry, provenance and resume-by-key.
  --session main : D (S=64 = equal forwards), T0, T (S=8, B=32) on all tiers; D at S=256 (Baseline A style) on a subset;
                   T grid S x B and D S-sweep at the Long tier; held-out masked CE (T at B in {16,32,64}); long-context CE D vs T;
                   judge NLL (Qwen int8) for every generation.
  --session q    : Q greedy on all tiers; Q held-out AR perplexity; Q judge-NLL of its own generations (biased; recorded as such).
  --session swap : T with the bf16 context tower (user's experiment): all tiers x runs; held-out masked CE.
  --session gsm  : GSM8K 200-item 4-shot exact match for D and T.   --session gsmq : same for Q."""
import argparse, json, re, sys, time
import numpy as np, torch
from twostory.util import REPO, gpu_telemetry
from bench.harness import MainSession, QSession, Recorder, load_prompt_set, session_header, rep_rate, BOS_DREAM

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True, choices=["main", "q", "swap", "swapce", "gsm", "gsmq"])
ap.add_argument("--runs", type=int, default=5); ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--tiers", default="short,medium,long,vlong,hand"); ap.add_argument("--limit", type=int, default=None, help="prompts per tier (smoke)")
ap.add_argument("--gen-tokens", type=int, default=256); ap.add_argument("--ckpt", default="checkpoints/phase4_run/best.pt")
ap.add_argument("--sweep-prompts", type=int, default=8); ap.add_argument("--sweep-runs", type=int, default=3)
ap.add_argument("--gsm-n", type=int, default=200); ap.add_argument("--out", default=None); ap.add_argument("--no-extras", action="store_true")
a = ap.parse_args()
out = REPO / "logs/phase5" / (a.out or f"raw_{a.session.removesuffix(chr(99)+chr(101))}.jsonl"); rec = Recorder(out)
tiers = a.tiers.split(","); prompts, ps = load_prompt_set(tiers, a.limit)
N = a.gen_tokens
print(f"session {a.session}: {len(prompts)} prompts, {a.runs} runs, {N} gen tokens -> {out}", flush=True)
rec.write({"key": f"header:{a.session}:{time.time()}", **session_header(a.session, {"args": vars(a)})})

def vram_guard(where, peak_gib=None):
    """The xe driver spills VRAM overflow into PINNED system RAM, which on this 15 GB box ends in a global OOM kill of the desktop
    (2026-09-20). Abort the session cleanly instead. rc=4 tells the driver not to restart it."""
    free = torch.xpu.mem_get_info()[0] / 2**30
    if free < 0.8 or (peak_gib is not None and peak_gib > 30.3):
        print(f"VRAM GUARD at {where}: device free {free:.2f} GiB, peak {peak_gib} GiB -> aborting session (rc=4)", flush=True); sys.exit(4)


def record(system, p, run, r, S=None, B=None, judge=None, tag=""):
    vram_guard(f"{system}:{p['id']}", r.get("peak_gib"))
    rec.write({"key": f"{tag}{system}:{p['id']}:{run}:S{S}:B{B}", "system": system, "prompt_id": p["id"], "tier": p["tier"], "run": run, "S": S, "B": B,
               "prompt_tokens": p["prompt_tokens"], "gen_tokens": len(r["ids"]), "total_s": r["total_s"], "ttft_s": r["ttft_s"], "prefill_s": r["prefill_s"],
               "steady_tok_s": r["steady_tok_s"], "wall_tok_s": len(r["ids"]) / r["total_s"], "forwards": r["forwards"], "peak_gib": r["peak_gib"],
               "per_block_s": r.get("per_block_s"), "rep": rep_rate(r["ids"]), "judge_nll": judge, "gpu": {k: gpu_telemetry()[k] for k in ("act_freq_mhz", "temp_max_c")},
               "ids": r["ids"]})

# ------------------------------------------------------------------------------------------------ main / swap
if a.session in ("main", "swap", "swapce"):
    bf16ctx = a.session != "main"
    ce_only = a.session == "swapce"   # held-out CE only: no judge needed, so drop Qwen's lm_head (-1.1 GiB) for headroom
    ses = MainSession(a.ckpt, context_bits=16 if bf16ctx else 8, keep_lm_head=not ce_only)
    if ce_only: prompts = []
    plan = [("T", 8, 32)] if bf16ctx else [("D", 64, None), ("T0", 8, 32), ("T", 8, 32), ("T", 32, 32)]  # T S=32 = 256 forwards, D's S=256 pairing
    tag = "bf16ctx:" if bf16ctx else ""
    names = [s for s, _, _ in plan]
    # warm-up each system
    w = prompts[0] if prompts else None
    for s, S, B in plan:
        if w is None: break
        ses.set_system(s)
        for _ in range(a.warmup): (ses.gen_D(w["ids"][:64], 32, 8) if s == "D" else ses.gen_T(w["ids"][:64], 32, 8, 32))
    print("warm-up done", flush=True)
    t_start = time.time(); n_done = 0
    for pi, p in enumerate(prompts):
        for run in range(a.runs):
            for s, S, B in plan:               # interleave systems within each (prompt, run)
                if S == 32 and run >= 3: continue   # the slow S=32 point: 3 repeats
                if rec.has(f"{tag}{('Tbf16' if bf16ctx else s)}:{p['id']}:{run}:S{S}:B{B}"): continue
                ses.set_system(s)
                r = ses.gen_D(p["ids"], N, S) if s == "D" else ses.gen_T(p["ids"], N, S, B)
                record("Tbf16" if bf16ctx else s, p, run, r, S, B, judge=ses.judge_nll(p["ids"], r["ids"]), tag=tag); n_done += 1
        el = time.time() - t_start
        print(f"[{pi+1}/{len(prompts)}] {p['id']} done; {el/60:.1f} min elapsed, ~{el/(pi+1)*(len(prompts)-pi-1)/60:.0f} min left", flush=True)
    if bf16ctx:
        for B in (32,):
            if not rec.has(f"heldout_ce:Tbf16:B{B}"):
                ses.set_system("T"); rec.write({"key": f"heldout_ce:Tbf16:B{B}", "kind": "heldout_ce", "system": "Tbf16", "B": B, "ce": ses.heldout_ce_T(B)})
                print("held-out CE (bf16 ctx):", rec.done and "recorded", flush=True)
    elif not a.no_extras:
        # D at S=256 (1 token/forward, Baseline A style) on a subset
        sub = [p for t in ("short", "medium", "long", "vlong") for p in [q for q in prompts if q["tier"] == t][:5]]
        for p in sub:
            for run in range(min(3, a.runs)):
                if rec.has(f"D:{p['id']}:{run}:S256:BNone"): continue
                ses.set_system("D"); r = ses.gen_D(p["ids"], N, 256); record("D", p, run, r, 256, None, judge=ses.judge_nll(p["ids"], r["ids"]))
            print(f"D S=256 {p['id']} done", flush=True)
        # sweep at the Long tier: T grid, D S-sweep
        longp = [p for p in prompts if p["tier"] == "long"][: a.sweep_prompts]
        grid = [(S, B) for S in (4, 8, 16, 32) for B in (16, 32, 64)]
        for p in longp:
            for run in range(a.sweep_runs):
                for S, B in grid:
                    if rec.has(f"sweep:T:{p['id']}:{run}:S{S}:B{B}"): continue
                    ses.set_system("T"); r = ses.gen_T(p["ids"], N, S, B); record("T", p, run, r, S, B, judge=ses.judge_nll(p["ids"], r["ids"]), tag="sweep:")
                for S in (32, 64, 128, 256):
                    if rec.has(f"sweep:D:{p['id']}:{run}:S{S}:BNone"): continue
                    ses.set_system("D"); r = ses.gen_D(p["ids"], N, S); record("D", p, run, r, S, None, judge=ses.judge_nll(p["ids"], r["ids"]), tag="sweep:")
            print(f"sweep {p['id']} done", flush=True)
        # held-out masked CE: T at B in {16,32,64}, T0, and long-context retention D vs T
        for B in (16, 32, 64):
            if not rec.has(f"heldout_ce:T:B{B}"):
                ses.set_system("T"); rec.write({"key": f"heldout_ce:T:B{B}", "kind": "heldout_ce", "system": "T", "B": B, "ce": ses.heldout_ce_T(B)})
        if not rec.has("heldout_ce:T0:B32"):
            ses.set_system("T0"); rec.write({"key": "heldout_ce:T0:B32", "kind": "heldout_ce", "system": "T0", "B": 32, "ce": ses.heldout_ce_T(32)})
        for ctx in (128, 512, 1024, 2048, 3968):
            for s in ("D", "T"):
                if not rec.has(f"longctx:{s}:{ctx}"):
                    rec.write({"key": f"longctx:{s}:{ctx}", "kind": "longctx_ce", "system": s, "ctx_len": ctx, "ce": ses.longctx_ce(s, ctx)})
            print(f"longctx {ctx} done", flush=True)

# ------------------------------------------------------------------------------------------------ Q
elif a.session == "q":
    ses = QSession(); w = prompts[0]
    for _ in range(a.warmup): ses.gen(w["ids"][:64], 32)
    t_start = time.time()
    for pi, p in enumerate(prompts):
        for run in range(a.runs):
            if rec.has(f"Q:{p['id']}:{run}:SNone:BNone"): continue
            r = ses.gen(p["ids"], N); record("Q", p, run, r, judge=ses.judge_nll(p["ids"], r["ids"]))
        el = time.time() - t_start; print(f"[{pi+1}/{len(prompts)}] {p['id']} done; ~{el/(pi+1)*(len(prompts)-pi-1)/60:.0f} min left", flush=True)
    if not rec.has("heldout_ppl:Q"):
        rec.write({"key": "heldout_ppl:Q", "kind": "heldout_ppl", "system": "Q", **ses.heldout_ppl()})

# ------------------------------------------------------------------------------------------------ GSM8K
else:
    import pyarrow.parquet as pq
    rows = pq.read_table(next((REPO / "data/gsm8k/main").glob("test-*.parquet"))).to_pylist()
    rng = np.random.default_rng(0); idx = rng.permutation(len(rows)); shots = [rows[i] for i in idx[:4]]; items = [rows[i] for i in idx[4:4 + a.gsm_n]]
    def fmt(q, ans=None): return f"Question: {q}\nAnswer: " + (ans.split('####')[0].strip() + f"\nThe answer is {ans.split('####')[1].strip()}.\n\n" if ans else "")
    prefix = "".join(fmt(s["question"], s["answer"]) for s in shots)
    def gold(ans): return ans.split("####")[1].strip().replace(",", "")
    def extract(text):
        m = re.findall(r"answer is\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", text)
        if not m: m = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
        return m[0].replace(",", "").rstrip(".") if m else None
    if a.session == "gsm":
        ses = MainSession(a.ckpt); systems = [("D", 64, None), ("T", 8, 32)]
    else:
        ses = QSession(); systems = [("Q", None, None)]
    tokq = ses.tokq
    for k, it in enumerate(items):
        ids = tokq(prefix + fmt(it["question"]), add_special_tokens=False).input_ids
        for s, S, B in systems:
            if rec.has(f"gsm:{s}:{k}"): continue
            if s == "Q": r = ses.gen(ids, N)
            else:
                ses.set_system(s); r = ses.gen_D(ids, N, S) if s == "D" else ses.gen_T(ids, N, S, B)
            text = tokq.decode(r["ids"]); text = text.split("\n\nQuestion:")[0]
            pred = extract(text); g = gold(it["answer"])
            rec.write({"key": f"gsm:{s}:{k}", "kind": "gsm", "system": s, "item": k, "S": S, "B": B, "gold": g, "pred": pred, "correct": pred is not None and abs(float(pred) - float(g)) < 1e-6 if pred not in (None, "") and g.replace("-","").replace(".","").isdigit() else pred == g,
                       "total_s": r["total_s"], "prompt_tokens": len(ids), "text": text[:600]})
        if (k + 1) % 10 == 0:
            acc = {s: np.mean([json.loads(l)["correct"] for l in open(out) if f'"key": "gsm:{s}:' in l]) for s, _, _ in systems}
            print(f"[{k+1}/{len(items)}] running accuracy {acc}", flush=True)
print("SESSION COMPLETE", flush=True)
