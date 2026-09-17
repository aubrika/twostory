"""Phase 3 / 30: pilot run. Train on ~N tokens with the real config; held-out eval + fixed generation samples every
`eval_every` steps; keep best checkpoint by held-out loss; early-stop on 3 consecutive held-out increases.
Gates (automated part): held-out loss decreasing; no NaN; samples free of mask tokens. Eyeballing the samples is yours."""
import argparse, json, time, math, numpy as np, torch
from twostory.util import gate, finish, write_report, MemTracker, load_tokenizer, REPO, provenance, gib, gpu_telemetry
from twostory.model import TwoStory, LoraSpec
from twostory.bridge import BridgeConfig
from twostory.train import Trainer, TrainConfig
from twostory.prompts import PROMPT_SHORT

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=float, default=8e6)
ap.add_argument("--batch-seqs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--eval-seqs", type=int, default=32)
ap.add_argument("--lora-rank", type=int, default=16)
ap.add_argument("--run-name", default="phase3_pilot")
ap.add_argument("--gen-tokens", type=int, default=64)
ap.add_argument("--resume", action="store_true", help="resume from checkpoints/<run>/last.pt if present")
a = ap.parse_args()

tokq = load_tokenizer("qwen")
pset = json.load(open(REPO / "data/prompt_set.json"))["prompts"]
sample_prompts = [("protocol_short", tokq(PROMPT_SHORT).input_ids)] + \
                 [(p["id"], p["ids"]) for p in pset if p["id"] in ("short_00", "short_07", "medium_03", "hand_story_open")]

with MemTracker() as mt:
    ts = TwoStory.load(BridgeConfig(), LoraSpec(rank=a.lora_rank))
print("load:", mt.summary(), ts.param_report(), flush=True)
steps = math.ceil(a.tokens / (1024 * a.batch_seqs))
cfg = TrainConfig(steps=steps, batch_seqs=a.batch_seqs, lr=a.lr, warmup=a.warmup, eval_every=a.eval_every, eval_seqs=a.eval_seqs,
                  run_name=a.run_name, early_stop_patience=3, log_every=1)
tr = Trainer(ts, cfg)
samples_f = open(tr.run_dir / "samples.jsonl", "a")
resumed = False
if a.resume and (tr.run_dir / "last.pt").exists():
    st = tr.load_trainable(tr.run_dir / "last.pt"); resumed = True
    print(f"RESUMED from step {tr.step} ({tr.tokens_seen} tokens), best eval {tr.best_eval}", flush=True)

def sample_all(tag):
    ts.eval(); ts.dream.gradient_checkpointing_disable()
    out = []
    for name, ids in sample_prompts:
        r = ts.generate_block(torch.tensor([ids], device="xpu"), max_new_tokens=a.gen_tokens, block_size=32, steps=8)
        out.append({"prompt": name, "ids": r["ids"], "text": tokq.decode(r["ids"]), "tok_s": round(len(r["ids"]) / r["total_seconds"], 1)})
    samples_f.write(json.dumps({"step": tr.step, "tag": tag, "samples": out}) + "\n"); samples_f.flush()
    ts.dream.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); ts.train()
    return out

print(f"pilot: {steps} steps x {a.batch_seqs} seqs = {steps*a.batch_seqs*1024/1e6:.1f}M tokens; eval every {a.eval_every}", flush=True)
if not resumed:
    ev0 = tr.evaluate(); print(f"eval@0: {ev0}", flush=True)
    s0 = sample_all("step0"); print("sample@0:", repr(s0[0]["text"][:120]), flush=True)
    evals = [ev0]
else:
    evals = [{"step": -1, "eval_loss": v} for v in tr.eval_history] or [tr.evaluate()]
t_start = time.time(); step_start = tr.step; last_ckpt = time.time(); stopped_early = False
while tr.step < steps:
    rec = tr.train_step()
    if tr.step % 50 == 0 or tr.step <= 5:
        el = time.time() - t_start; eta = el / max(1, tr.step - step_start) * (steps - tr.step)
        print(f"step {tr.step:5d}/{steps} loss {rec['loss']:.4f} lr {rec['lr']:.2e} gn {rec['grad_norm']:.2f} {rec['seconds']:.2f}s "
              f"{rec['seq_tok_s']:.0f} tok/s vram {rec['vram_alloc_gib']}/{rec['vram_peak_gib']} GiB free {rec['dev_free_gib']} "
              f"gpu {rec.get('gpu',{}).get('act_freq_mhz')}MHz {rec.get('gpu',{}).get('temp_max_c')}C  eta {eta/3600:.2f}h", flush=True)
    if not math.isfinite(rec["loss"]):
        print("NaN loss - stopping"); break
    if tr.step % a.eval_every == 0 or tr.step == steps:
        ev = tr.evaluate(); evals.append(ev)
        smp = sample_all(f"step{tr.step}")
        print(f"eval@{tr.step}: loss {ev['eval_loss']:.4f} (ce {ev['eval_ce_masked']:.4f}) best {min(e['eval_loss'] for e in evals):.4f}  "
              f"sample: {repr(smp[0]['text'][:100])}", flush=True)
        if ev["eval_loss"] < tr.best_eval:
            tr.best_eval = ev["eval_loss"]; tr.save("best"); print("  saved best", flush=True)
        if tr.should_early_stop():
            print("EARLY STOP: held-out loss rose 3 evals in a row", flush=True); stopped_early = True; break
    if time.time() - last_ckpt > cfg.ckpt_every_sec:
        tr.save("last"); last_ckpt = time.time(); print(f"  saved last @ step {tr.step}", flush=True)
tr.save("last")
losses = [e["eval_loss"] for e in evals]
x = np.arange(len(losses)); slope = float(np.polyfit(x, losses, 1)[0]) if len(losses) > 2 else float("nan")
train_losses = [json.loads(l)["loss"] for l in open(tr.log_path) if '"loss"' in l and '"eval"' not in l]
rep = {"args": vars(a), "provenance": provenance(), "params": ts.param_report(), "steps_done": tr.step, "tokens_seen": tr.tokens_seen,
       "hours": round((time.time() - t_start) / 3600, 2), "evals": evals, "eval_slope_per_eval": slope, "stopped_early": stopped_early,
       "best_eval": tr.best_eval, "train_loss_first100_mean": float(np.mean(train_losses[:100])), "train_loss_last100_mean": float(np.mean(train_losses[-100:])),
       "peak_vram_gib": gib(torch.xpu.max_memory_allocated()), "gpu_end": gpu_telemetry(), "samples_file": str(tr.run_dir / "samples.jsonl")}
gate("held-out loss decreased (last < first)", losses[-1] < losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}")
gate("held-out trend is downward (linear slope < 0)", slope < 0, f"slope {slope:.4f}/eval")
gate("best held-out < first by >= 10%", tr.best_eval < 0.9 * losses[0], f"best {tr.best_eval:.4f}")
gate("no NaN in training", all(math.isfinite(v) for v in train_losses))
gate("no early stop triggered", not stopped_early)
last = json.loads(open(tr.run_dir / "samples.jsonl").readlines()[-1])
gate("final samples contain no mask tokens", all(ts.mask_id not in s["ids"] for s in last["samples"]))
write_report("30_pilot", rep, subdir="phase3")
finish("Phase 3 / 30 pilot")
