"""Phase 3 / 31: throughput. Bridged block-mode generation (S=8, B=32) vs plain Dream at prompt lengths short/512/2k,
128 new tokens, bridges cast to bf16, warm-up then median of N runs. Dream is measured at steps=128 (Baseline A's
setting, 1 token/forward) AND steps=32 (same number of forwards as the bridged run) so the comparison is honest both ways.
Gate (protocol): bridged faster than Baseline A on prompts >= 512 tokens."""
import argparse, json, time, statistics, torch
from twostory.util import gate, finish, write_report, MemTracker, load_tokenizer, REPO, provenance, gpu_telemetry
from twostory.model import TwoStory, LoraSpec
from twostory.bridge import BridgeConfig
from twostory.train import Trainer, TrainConfig
from twostory.prompts import PROMPT_SHORT, build_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None); ap.add_argument("--runs", type=int, default=5); ap.add_argument("--new-tokens", type=int, default=128)
ap.add_argument("--block-size", type=int, default=32); ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--prompts", default="short,p512,p2k"); a = ap.parse_args()

tokq, tokd = load_tokenizer("qwen"), load_tokenizer("dream")
ts = TwoStory.load(BridgeConfig(), LoraSpec(rank=16))
if a.ckpt:
    Trainer(ts, TrainConfig(run_name="throughput_tmp")).load_trainable(a.ckpt, load_optimizer=False)
ts.eval(); ts.bridges_to(torch.bfloat16)
prompts = {n: (PROMPT_SHORT if n == "short" else build_prompt(tokq, {"p512": 512, "p2k": 2048}[n])) for n in a.prompts.split(",")}

def set_lora(on: bool):
    from peft.tuners.tuners_utils import BaseTunerLayer
    for mod in ts.dream.modules():
        if isinstance(mod, BaseTunerLayer):
            mod.enable_adapters(on)

def med(fn, runs):
    fn(); vals = []
    for _ in range(runs):
        torch.xpu.synchronize(); t0 = time.perf_counter(); r = fn(); torch.xpu.synchronize(); vals.append((time.perf_counter() - t0, r))
    secs = sorted(v[0] for v in vals); q = statistics.quantiles(secs, n=4) if len(secs) >= 4 else [secs[0], secs[len(secs)//2], secs[-1]]
    return {"median_s": statistics.median(secs), "iqr_s": q[2] - q[0], "runs": len(secs)}, vals[-1][1]

rep = {"provenance": provenance(), "ckpt": a.ckpt, "new_tokens": a.new_tokens, "block_size": a.block_size, "steps": a.steps, "results": {}}
for name, text in prompts.items():
    idq = tokq(text, return_tensors="pt").input_ids.to("xpu"); idd = tokd(text, return_tensors="pt").input_ids.to("xpu")
    r = {"prompt_tokens": idq.shape[1], "gpu_before": {k: gpu_telemetry()[k] for k in ("act_freq_mhz", "temp_max_c")}}
    ts.dream._ambient_ctx = (None, None)
    set_lora(False)   # plain Dream: no bridge (no ambient ctx) AND no LoRA adapters
    for S in (128, 32):
        m, out = med(lambda: ts.dream.diffusion_generate(idd, max_new_tokens=a.new_tokens, steps=S, temperature=0.0, alg="entropy", alg_temp=None), a.runs)
        r[f"dream_S{S}"] = m | {"tok_s": round(a.new_tokens / m["median_s"], 2), "text": tokd.decode(out[0, idd.shape[1]:])[:200]}
    set_lora(True)
    m, out = med(lambda: ts.generate_block(idq, max_new_tokens=a.new_tokens, block_size=a.block_size, steps=a.steps, stop_at_eos=False), a.runs)
    r["bridged"] = m | {"tok_s": round(a.new_tokens / m["median_s"], 2), "ttft_s": out["ttft_seconds"], "prefill_s": out["prefill_seconds"],
                        "per_block_s": [round(b["seconds"], 3) for b in out["blocks"]], "text": tokq.decode(out["ids"])[:200]}
    rep["results"][name] = r
    print(f"[{name}] {idq.shape[1]} tok: Dream S=128 {r['dream_S128']['tok_s']} tok/s | Dream S=32 {r['dream_S32']['tok_s']} tok/s | "
          f"bridged S={a.steps},B={a.block_size} {r['bridged']['tok_s']} tok/s (ttft {r['bridged']['ttft_s']:.2f}s)", flush=True)
for name in prompts:
    if name != "short":
        rr = rep["results"][name]
        gate(f"[{name}] bridged faster than Baseline A (Dream S=128)", rr["bridged"]["tok_s"] > rr["dream_S128"]["tok_s"], f"{rr['bridged']['tok_s']} vs {rr['dream_S128']['tok_s']}")
        gate(f"[{name}] bridged faster than Dream at equal forwards (S=32)", rr["bridged"]["tok_s"] > rr["dream_S32"]["tok_s"], f"{rr['bridged']['tok_s']} vs {rr['dream_S32']['tok_s']}")
tag = ("trained" if a.ckpt else "T0") + "_plainD"
write_report(f"31_throughput_{tag}", rep, subdir="phase3")
finish(f"Phase 3 / 31 throughput ({tag})")
