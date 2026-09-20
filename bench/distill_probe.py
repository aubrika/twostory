"""Distillation feasibility probe (2026-09-20). Teacher = plain Dream decoded BLOCK-WISE (prompt + committed blocks in its own input,
32-token canvas, same confidence schedule as the bridged model) = the quality ceiling for this design and a target the student can
in principle compute. Student = trained bridge (context only via Qwen's cache).
  1. Ceiling: block-wise Dream on GSM8K (S=8: 200 items, S=32: 100 items) and on a prompt-set subset (repetition, judge NLL).
  2. Agreement: per denoising step, student vs teacher argmax / KL on (A) the teacher's own trajectory with a clean cache and
     (B) the student's own rollout (teacher queried on the student's states), broken down by block index -> capacity gap vs exposure bias."""
import argparse, json, re, sys, time
import numpy as np, torch, torch.nn.functional as F
from twostory.util import REPO
from twostory.dream import sample_tokens
from bench.harness import MainSession, Recorder, load_prompt_set, session_header, rep_rate, BOS_DREAM

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="checkpoints/phase4_run/best.pt")
ap.add_argument("--gsm-n", type=int, default=200); ap.add_argument("--gsm-n-s32", type=int, default=100); ap.add_argument("--agree-prompts", type=int, default=20)
a = ap.parse_args()
rec = Recorder(REPO / "logs/phase5/raw_probe.jsonl"); rec.write({"key": f"header:probe:{time.time()}", **session_header("probe", {"args": vars(a)})})
ses = MainSession(a.ckpt); ts = ses.ts; tokq = ses.tokq; MASK = ts.mask_id; B = 32; dev = "xpu"
ses._bridges(True); ts.dream._ambient_ctx = (None, None)

def guard(where):
    free = torch.xpu.mem_get_info()[0] / 2**30
    if free < 0.8: print(f"VRAM GUARD at {where}: free {free:.2f} GiB", flush=True); sys.exit(4)

@torch.no_grad()
def d_logits(ctx_ids, xb):
    """Teacher: plain Dream (LoRA off, no bridge) over [BOS] + context + block; logits for the block positions (hidden p-1 predicts p)."""
    ses._lora(False)
    x = torch.cat([torch.tensor([BOS_DREAM] + ctx_ids, device=dev), xb])[None]
    lg = ts.dream(x).logits[0]; n = 1 + len(ctx_ids)
    return lg[n - 1: n - 1 + len(xb)].float()

@torch.no_grad()
def t_logits(ctx_ids, xb):
    """Student: bridged Dream over [anchor, block] with the context only in Qwen's cache (cache must be built for ctx_ids)."""
    ses._lora(True); T = len(ctx_ids)
    x = torch.cat([torch.tensor([ctx_ids[-1]], device=dev), xb])[None]; pos = torch.arange(T - 1, T + len(xb), device=dev)[None]
    lg = ts.dream(x, position_ids=pos, ctx_kv=ts.ctx.kv()).logits[0]
    return lg[:-1].float()

def commit(xb, logits, i, S, eps=1e-3):
    """Dream's entropy-confidence schedule on one block: returns (new block, committed indices, argmax for every position)."""
    tt = torch.linspace(1, eps, S + 1); t, s = float(tt[i]), float(tt[i + 1])
    mask = xb == MASK; n_mask = int(mask.sum()); am = logits.argmax(-1)
    if n_mask == 0: return xb, torch.empty(0, dtype=torch.long, device=dev), am
    conf, x0 = sample_tokens(logits[mask], temperature=0.0, neg_entropy=True)
    n_tr = int(n_mask * (1 - s / t)) if i < S - 1 else n_mask
    full = torch.full((len(xb),), -torch.inf, device=dev); full[mask] = conf.float()
    idx = torch.topk(full, n_tr).indices if n_tr > 0 else torch.empty(0, dtype=torch.long, device=dev)
    cand = torch.full_like(xb, MASK); cand[mask] = x0; out = xb.clone(); out[idx] = cand[idx]
    return out, idx, am

@torch.no_grad()
def gen_blockwise_D(prompt_ids, n_new, S, stop_text=None):
    ctx = list(prompt_ids); new = []
    while len(new) < n_new:
        xb = torch.full((B,), MASK, device=dev, dtype=torch.long)
        for i in range(S): xb, _, _ = commit(xb, d_logits(ctx, xb), i, S)
        blk = xb.tolist(); new += blk; ctx += blk
        if stop_text and stop_text in tokq.decode(new): break
        if ts.eos_id in blk: break
    return new[:n_new]

# ---------------------------------------------------------------- preflight: exercise every code path on tiny inputs before the long loops
pf = tokq("The capital of France is", add_special_tokens=False).input_ids * 6
_ = gen_blockwise_D(pf, 32, 2); ts.ctx.build(torch.tensor([pf], device=dev)); xb0 = torch.full((B,), MASK, device=dev, dtype=torch.long)
_ = commit(xb0, t_logits(pf, xb0), 0, 2); _ = ses.judge_nll(pf, _[0].tolist()); print("preflight ok", flush=True)

# ---------------------------------------------------------------- 2. agreement probe first (short, the most informative)
prompts, _ps = load_prompt_set(["medium"]); S, NB = 8, 4
def metrics(lgD, lgT, xb, idxD):
    m = xb == MASK
    if not m.any(): return None
    aD, aT = lgD.argmax(-1), lgT.argmax(-1); pD = F.log_softmax(lgD[m], -1); pT = F.log_softmax(lgT[m], -1)
    return {"agree_masked": float((aD[m] == aT[m]).float().mean()), "agree_commit": float((aD[idxD] == aT[idxD]).float().mean()) if len(idxD) else None,
            "kl_D_T": float((pD.exp() * (pD - pT)).sum(-1).mean()), "n_masked": int(m.sum()), "top1_prob_D": float(pD.max(-1).values.exp().mean()), "top1_prob_T": float(pT.max(-1).values.exp().mean())}
for pi, p in enumerate(prompts[: a.agree_prompts]):
    for regime in ("A_teacher_states", "B_student_states"):
        if rec.has(f"agree:{regime}:{p['id']}"): continue
        ctx = list(p["ids"]); steps = []; text_blocks = []
        for b in range(NB):
            ts.ctx.build(torch.tensor([ctx], device=dev)); xb = torch.full((B,), MASK, device=dev, dtype=torch.long)
            for i in range(S):
                lgD, lgT = d_logits(ctx, xb), t_logits(ctx, xb)
                xD, idxD, _ = commit(xb, lgD, i, S); xT, idxT, _ = commit(xb, lgT, i, S)
                mm = metrics(lgD, lgT, xb, idxD)
                if mm: steps.append({"block": b, "step": i, **mm})
                xb = xD if regime.startswith("A") else xT
            blk = xb.tolist(); ctx += blk; text_blocks.append({"block": b, "rep": rep_rate(blk), "text": tokq.decode(blk)})
        rec.write({"key": f"agree:{regime}:{p['id']}", "kind": "agree", "regime": regime, "prompt_id": p["id"], "steps": steps, "blocks": text_blocks}); guard("agree")
    print(f"[agree {pi+1}/{a.agree_prompts}] {p['id']} done", flush=True)

# ---------------------------------------------------------------- 1a. block-wise Dream on a prompt-set subset (S=8): repetition + judge
allp, _ = load_prompt_set(["short", "medium", "long"])
for tier in ("short", "medium", "long"):
    for p in [q for q in allp if q["tier"] == tier][:10]:
        if rec.has(f"dblk:{p['id']}:S8"): continue
        t0 = time.time(); g = gen_blockwise_D(p["ids"], 256, 8)
        rec.write({"key": f"dblk:{p['id']}:S8", "kind": "dblk_gen", "tier": tier, "prompt_id": p["id"], "S": 8, "rep": rep_rate(g), "judge_nll": ses.judge_nll(p["ids"], g), "seconds": time.time() - t0, "ids": g}); guard("dblk")
    print(f"[dblk prompts] {tier} done", flush=True)

# ---------------------------------------------------------------- 1b. block-wise Dream on GSM8K (same items / prompt as the gsm sessions)
import pyarrow.parquet as pq
rows = pq.read_table(next((REPO / "data/gsm8k/main").glob("test-*.parquet"))).to_pylist()
rng = np.random.default_rng(0); idx = rng.permutation(len(rows)); shots = [rows[i] for i in idx[:4]]; items = [rows[i] for i in idx[4:4 + a.gsm_n]]
def fmt(q, ans=None): return f"Question: {q}\nAnswer: " + (ans.split('####')[0].strip() + f"\nThe answer is {ans.split('####')[1].strip()}.\n\n" if ans else "")
prefix = "".join(fmt(s_["question"], s_["answer"]) for s_ in shots)
def extract(text):
    m = re.findall(r"answer is\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", text)
    strict = m[0].replace(",", "").rstrip(".") if m else None
    m2 = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
    return strict, (strict or (m2[0].replace(",", "").rstrip(".") if m2 else None))
for S_, n_items in ((8, a.gsm_n), (32, a.gsm_n_s32)):
    for k, it in enumerate(items[:n_items]):
        if rec.has(f"gsm:Dblk{S_}:{k}"): continue
        ids = tokq(prefix + fmt(it["question"]), add_special_tokens=False).input_ids
        g = gen_blockwise_D(ids, 256, S_, stop_text="\n\nQuestion:"); text = tokq.decode(g).split("\n\nQuestion:")[0]
        strict, loose = extract(text); gold = it["answer"].split("####")[1].strip().replace(",", "")
        def ok(pv):
            try: return pv is not None and abs(float(pv) - float(gold)) < 1e-6
            except ValueError: return pv == gold
        rec.write({"key": f"gsm:Dblk{S_}:{k}", "kind": "gsm", "system": f"Dblk S={S_}", "item": k, "gold": gold, "pred": loose, "correct": ok(loose), "correct_strict": ok(strict), "reached_answer": strict is not None, "text": text[:600]}); guard("gsm")
        if (k + 1) % 20 == 0:
            R = [json.loads(l) for l in open(rec.path) if f'"key": "gsm:Dblk{S_}:' in l]
            print(f"[gsm Dblk S={S_} {k+1}/{n_items}] acc {np.mean([r['correct'] for r in R]):.3f} strict {np.mean([r['correct_strict'] for r in R]):.3f} reached {np.mean([r['reached_answer'] for r in R]):.2f}", flush=True)
print("SESSION COMPLETE", flush=True)
