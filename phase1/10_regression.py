"""Phase 1 / 10: THE regression gate. Bridges attached (zero-init out_proj), LoRA injected (B=0), context cache
built and live; Dream's own diffusion loop must reproduce Baseline A token-for-token. Then prove the wiring is
real: perturb one bridge's out_proj and the output must change."""
import json, time, torch
from twostory.util import gate, finish, write_report, MemTracker, load_tokenizer, REPO, DENOISER_BITS, provenance
from twostory.model import TwoStory, LoraSpec
from twostory.bridge import BridgeConfig
from twostory.prompts import PROMPT_SHORT, build_prompt

base = json.load(open(REPO / "logs/phase0" / f"baseline_A_dream_{DENOISER_BITS}bit.json"))
tok = load_tokenizer("dream")
with MemTracker() as mt:
    ts = TwoStory.load(BridgeConfig(), LoraSpec())
print("load:", mt.summary()); print("params:", ts.param_report())
gen_kw = dict(max_new_tokens=base["new_tokens"], steps=base["steps"], temperature=base["temperature"], alg=base["alg"],
              alg_temp=None, return_dict_in_generate=True)
rep = {"provenance": provenance(), "params": ts.param_report(), "prompts": {}}
for name, b in base["prompts"].items():
    prompt = PROMPT_SHORT if name == "short" else build_prompt(tok, b["prompt_tokens"] if name != "short" else 0)
    ids = tok(prompt, return_tensors="pt").input_ids.to("xpu")
    assert ids.shape[1] == b["prompt_tokens"], (ids.shape, b["prompt_tokens"])
    calls0 = sum(br.calls for br in ts.bridges)
    torch.xpu.synchronize(); t0 = time.time()
    out = ts.generate_full(ids, **gen_kw)
    torch.xpu.synchronize(); dt = time.time() - t0
    got = out.sequences[0, ids.shape[1]:].tolist()
    calls = sum(br.calls for br in ts.bridges) - calls0
    same = got == b["ids"]
    first_diff = next((i for i, (x, y) in enumerate(zip(got, b["ids"])) if x != y), None)
    rep["prompts"][name] = {"identical": same, "first_diff": first_diff, "bridge_calls": calls, "seconds": round(dt, 2),
                            "cache_tokens": ts.ctx.length, "cache_mib": round(ts.ctx.bytes() / 2**20, 1)}
    print(f"[{name}] identical={same} first_diff={first_diff} bridge calls={calls} {dt:.1f}s cache={ts.ctx.length} tok / {ts.ctx.bytes()/2**20:.0f} MiB")
    gate(f"[{name}] token-identical to Baseline A", same, f"first diff at {first_diff}")
    gate(f"[{name}] bridges executed on every forward", calls == len(ts.bridges) * base["steps"], f"{calls} calls")
    gate(f"[{name}] cache length == prompt length", ts.ctx.length == ids.shape[1])

# wiring proof: a non-zero bridge must change the output
br = ts.bridges[14]
with torch.no_grad():
    br.out_proj.weight.normal_(0, 0.02)
ids = tok(PROMPT_SHORT, return_tensors="pt").input_ids.to("xpu")
got = ts.generate_full(ids, **gen_kw).sequences[0, ids.shape[1]:].tolist()
changed = got != base["prompts"]["short"]["ids"]
with torch.no_grad():
    br.out_proj.weight.zero_()
got2 = ts.generate_full(ids, **gen_kw).sequences[0, ids.shape[1]:].tolist()
gate("perturbed bridge changes output (wiring is live)", changed)
gate("restored bridge is identical again", got2 == base["prompts"]["short"]["ids"])
rep["wiring_proof"] = {"changed": changed, "restored_identical": got2 == base["prompts"]["short"]["ids"]}
write_report("10_regression", rep, subdir="phase1")
finish("Phase 1 / 10 regression")
