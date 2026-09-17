"""Daily check-in for the Phase 4 run: progress, held-out curve, gap closed, health, latest samples. CPU only; safe while training."""
import json, os, subprocess, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
run = REPO / "checkpoints" / (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUN", "phase4_run"))
if not (run / "train.jsonl").exists(): sys.exit(f"no run at {run}")
cfg = json.load(open(run / "config.json"))["train"]
steps, evals = [], []
for l in open(run / "train.jsonl"):
    d = json.loads(l); (evals if "eval" in d else steps).append(d.get("eval", d))
plan = cfg["steps"]; last = steps[-1] if steps else {"step": 0, "tokens_seen": 0, "time": time.time()}
running = subprocess.run(["pgrep", "-f", f"phase4/40_run.py --run-name {run.name}"], capture_output=True).returncode == 0
print(f"=== {run.name}: {'RUNNING' if running else 'stopped'}  |  step {last['step']}/{plan} ({100*last['step']/plan:.1f}%)  |  {last['tokens_seen']/1e6:.1f}M of {plan*cfg['batch_seqs']*1024/1e6:.0f}M tokens")
if len(steps) > 200:
    recent = steps[-200:]; rate = sum(r["seq_tok_s"] for r in recent) / len(recent)
    print(f"throughput {rate:.0f} train tok/s  |  remaining at this rate: {(plan-last['step'])*cfg['batch_seqs']*1024/rate/3600:.1f} h of training time")
    print(f"train loss: first-200 mean {sum(r['loss'] for r in steps[:200])/200:.4f}  ->  last-200 mean {sum(r['loss'] for r in recent)/200:.4f}   lr now {last['lr']:.2e}")
    print(f"health: min device-free {min(r['dev_free_gib'] for r in steps):.2f} GiB | max reserved {max(r['vram_reserved_gib'] for r in steps):.2f} GiB | "
          f"max temp {max(r['gpu']['temp_max_c'] for r in steps if r.get('gpu'))} C | min clock {min(r['gpu']['act_freq_mhz'] for r in steps if r.get('gpu') and r['gpu']['act_freq_mhz'])} MHz")
wd = REPO / "logs/phase4" / f"{run.name}.watchdog.log"
if wd.exists():
    w = open(wd).read(); print(f"watchdog: {w.count('launching segment')} launches, {w.count('STALE')} wedge-restarts, {w.count('ended cleanly')} clean segment ends")
print("\nheld-out evals (step: loss / token-weighted CE / gap closed toward plain Dream):")
for e in evals[-12:]:
    print(f"  {e['step']:6d}: {e['eval_loss']:.4f} / {e.get('eval_ce_token_weighted', float('nan')):.4f} / {100*e['gap_closed']:.1f}%" if "gap_closed" in e else f"  {e['step']:6d}: {e['eval_loss']:.4f}")
if len(evals) >= 2:
    best = min(evals, key=lambda e: e["eval_loss"]); rising = sum(1 for a, b in zip(evals[-4:], evals[-3:]) if b["eval_loss"] > a["eval_loss"])
    print(f"best {best['eval_loss']:.4f} @ step {best['step']}; last 3 intervals rising: {rising}/3 (early-stop rule fires at 3/3)")
for f in sorted(run.glob("segment_*.json")):
    s = json.load(open(f)); print(f"segment {s['segment']}: {s['hours']} h, ended at step {s['step']} - {s['reason']}")
if (run / "samples.jsonl").exists():
    rows = [json.loads(l) for l in open(run / "samples.jsonl")]
    print(f"\nlatest samples (step {rows[-1]['step']}, {rows[-1].get('sample_steps')} denoising steps per 32-token block):")
    for smp in rows[-1]["samples"]: print(f"  [{smp['prompt']}] {smp['text'][:200]!r}")
