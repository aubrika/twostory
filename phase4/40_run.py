"""Phase 4: the run, piecewise. Trains toward --total-tokens under a warmup-stable-decay schedule, in segments of at most
--max-hours. Stops cleanly (checkpoint incl. optimizer, exit 0) when: the segment time is up, checkpoints/<run>/STOP exists
(phase4/stop.sh), SIGTERM arrives, the plan is complete, or held-out loss rose 3 evals in a row. Resume is exact:
data order, context lengths and masks are pure functions of (seed, step). Exit != 0 only on a crash (watchdog restarts)."""
import argparse, json, math, os, signal, sys, time, numpy as np, torch
from twostory.util import MemTracker, load_tokenizer, REPO, provenance, gib, gpu_telemetry
from twostory.model import TwoStory, LoraSpec
from twostory.bridge import BridgeConfig
from twostory.train import Trainer, TrainConfig
from twostory.prompts import PROMPT_SHORT

ap = argparse.ArgumentParser()
ap.add_argument("--run-name", default="phase4_run")
ap.add_argument("--total-tokens", type=float, default=100e6, help="planned total; sets where the lr decay begins. May be RAISED later, before decay starts.")
ap.add_argument("--max-hours", type=float, default=20.0, help="segment length; the run stops itself after this long")
ap.add_argument("--max-steps-this-segment", type=int, default=0, help="testing aid")
ap.add_argument("--init-from", default="checkpoints/phase3_pilot/best.pt", help="warm-start weights (no optimizer) when no last.pt exists; '' for cold start")
ap.add_argument("--batch-seqs", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=200); ap.add_argument("--decay-frac", type=float, default=0.15)
ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--sample-steps", type=int, default=16)
ap.add_argument("--gen-tokens", type=int, default=64); ap.add_argument("--seed", type=int, default=4)
a = ap.parse_args()

stop = {"flag": False, "reason": None}
signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True, reason="SIGTERM"))
tokq = load_tokenizer("qwen")
pset = json.load(open(REPO / "data/prompt_set.json"))["prompts"]
sample_prompts = [("protocol_short", tokq(PROMPT_SHORT).input_ids)] + [(p["id"], p["ids"]) for p in pset if p["id"] in ("short_00", "short_07", "medium_03", "hand_story_open")]

ts = TwoStory.load(BridgeConfig(), LoraSpec(rank=16))
steps = math.ceil(a.total_tokens / (1024 * a.batch_seqs))
cfg = TrainConfig(steps=steps, batch_seqs=a.batch_seqs, lr=a.lr, warmup=a.warmup, schedule="wsd", decay_frac=a.decay_frac, eval_every=a.eval_every,
                  eval_seqs=32, run_name=a.run_name, early_stop_patience=3, log_every=1, seed=a.seed, stateless_sampling=True, optimizer="adamw8bit")
tr = Trainer(ts, cfg)
run = tr.run_dir; stop_file = run / "STOP"; state_file = run / "state.json"
if stop_file.exists(): stop_file.unlink()
if (run / "last.pt").exists():
    tr.load_trainable(run / "last.pt"); print(f"RESUMED at step {tr.step} ({tr.tokens_seen/1e6:.1f}M tokens), best eval {tr.best_eval:.4f}", flush=True)
elif a.init_from:
    tr.load_trainable(REPO / a.init_from, load_optimizer=False); tr.step = 0; tr.tokens_seen = 0; tr.eval_history = []; tr.best_eval = math.inf
    print(f"WARM START from {a.init_from} (weights only; fresh optimizer, step 0)", flush=True)
samples_f = open(run / "samples.jsonl", "a")

def sample_all():
    ts.eval(); ts.dream.gradient_checkpointing_disable(); out = []
    for name, ids in sample_prompts:
        r = ts.generate_block(torch.tensor([ids], device="xpu"), max_new_tokens=a.gen_tokens, block_size=32, steps=a.sample_steps)
        out.append({"prompt": name, "ids": r["ids"], "text": tokq.decode(r["ids"])})
    samples_f.write(json.dumps({"step": tr.step, "sample_steps": a.sample_steps, "samples": out}) + "\n"); samples_f.flush()
    ts.dream.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); ts.train()
    return out

def do_eval():
    ev = tr.evaluate(); smp = sample_all()
    gc = f" gap-closed {100*ev['gap_closed']:.1f}%" if "gap_closed" in ev else ""
    print(f"eval@{tr.step}: loss {ev['eval_loss']:.4f} ce {ev['eval_ce_token_weighted']:.4f}{gc} best {min(tr.best_eval, ev['eval_loss']):.4f}  sample: {smp[0]['text'][:90]!r}", flush=True)
    if ev["eval_loss"] < tr.best_eval:
        tr.best_eval = ev["eval_loss"]; tr.save("best"); print("  saved best", flush=True)
    return ev

seg_start, seg_step0, last_ckpt = time.time(), tr.step, time.time()
print(f"plan: {steps} steps = {steps*a.batch_seqs*1024/1e6:.0f}M tokens (wsd: lr {a.lr:g} flat until step {int(steps*(1-a.decay_frac))}); "
      f"segment: up to {a.max_hours} h from step {tr.step}; eval every {a.eval_every}", flush=True)
if tr.step == 0 or not tr.eval_history: do_eval()
reason = "plan complete"
while tr.step < steps:
    rec = tr.train_step()
    if not math.isfinite(rec["loss"]): reason = "NaN loss"; print("NaN loss - stopping", flush=True); tr.save("nan_dump"); sys.exit(3)
    if tr.step % 100 == 0 or tr.step - seg_step0 <= 3:
        el = time.time() - seg_start; rate = (tr.step - seg_step0) / el
        print(f"step {tr.step:6d}/{steps} ({100*tr.step/steps:.1f}%) loss {rec['loss']:.4f} lr {rec['lr']:.2e} gn {rec['grad_norm']:.2f} {rec['seconds']:.2f}s "
              f"{rec['seq_tok_s']:.0f} tok/s free {rec['dev_free_gib']} GiB {rec['gpu'].get('act_freq_mhz')}MHz {rec['gpu'].get('temp_max_c')}C  "
              f"seg {el/3600:.2f}/{a.max_hours}h  plan-eta {(steps-tr.step)/rate/3600:.1f}h", flush=True)
    if tr.step % a.eval_every == 0:
        do_eval()
        if tr.should_early_stop(): reason = "EARLY STOP: held-out loss rose 3 evals in a row"; break
    if time.time() - last_ckpt > cfg.ckpt_every_sec:
        tr.save("last"); last_ckpt = time.time()
    if stop_file.exists(): stop.update(flag=True, reason="STOP file")
    if stop["flag"]: reason = f"stop requested ({stop['reason']})"; break
    if time.time() - seg_start > a.max_hours * 3600: reason = f"segment time limit ({a.max_hours} h)"; break
    if a.max_steps_this_segment and tr.step - seg_step0 >= a.max_steps_this_segment: reason = "segment step limit (test)"; break
if tr.step >= steps and (not tr.eval_history or tr.step % a.eval_every): do_eval()
tr.save("last")
seg_n = len(list(run.glob("segment_*.json"))) + 1
summary = {"segment": seg_n, "reason": reason, "step": tr.step, "plan_steps": steps, "tokens_seen": tr.tokens_seen, "hours": round((time.time() - seg_start) / 3600, 2),
           "best_eval": tr.best_eval, "eval_history": tr.eval_history, "peak_vram_gib": gib(torch.xpu.max_memory_allocated()),
           "provenance": provenance(), "gpu_end": gpu_telemetry(), "finished": tr.step >= steps or reason.startswith("EARLY")}
(run / f"segment_{seg_n}.json").write_text(json.dumps(summary, indent=2, default=str)); state_file.write_text(json.dumps(summary, indent=2, default=str))
if stop_file.exists(): stop_file.unlink()
print(f"SEGMENT {seg_n} END: {reason}; step {tr.step}/{steps}, {tr.tokens_seen/1e6:.1f}M tokens, best eval {tr.best_eval:.4f}", flush=True)
