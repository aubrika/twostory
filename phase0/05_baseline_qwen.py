"""Phase 0 / 05: Baseline B. Qwen2.5-7B autoregressive greedy generation with tokens/sec."""
import argparse, time, torch
from twostory.util import gate, finish, write_report, MemTracker, load_qwen, load_tokenizer
from twostory.prompts import PROMPT_SHORT, build_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--bits", type=int, default=8)
ap.add_argument("--new-tokens", type=int, default=128)
ap.add_argument("--prompts", default="short,p512")
a = ap.parse_args()

tok = load_tokenizer("qwen")
with MemTracker() as mt:
    model = load_qwen(a.bits)
print("load:", mt.summary())

def gen(prompt: str, new_tokens: int):
    ids = tok(prompt, return_tensors="pt").input_ids.to("xpu")
    torch.xpu.synchronize(); t0 = time.time()
    out = model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    torch.xpu.synchronize(); dt = time.time() - t0
    return ids.shape[1], out[0, ids.shape[1]:].tolist(), dt

print("warmup..."); gen(PROMPT_SHORT, 16)
rep = {"bits": a.bits, "new_tokens": a.new_tokens, "greedy": True, "prompts": {}}
for name in a.prompts.split(","):
    prompt = PROMPT_SHORT if name == "short" else build_prompt(tok, int(name[1:]) if name[1:].isdigit() else {"p2k": 2048}[name])
    plen, s1, t1 = gen(prompt, a.new_tokens)
    with MemTracker() as gm:
        _, s2, t2 = gen(prompt, a.new_tokens)
    text = tok.decode(s2); tok_s = round(a.new_tokens / t2, 2)
    rep["prompts"][name] = {"prompt_tokens": plen, "ids": s2, "text": text, "seconds": round(t2, 2), "tok_s": tok_s,
                           "deterministic": s1 == s2, "gen_mem": gm.summary()}
    print(f"\n=== [{name}] prompt {plen} tok -> {a.new_tokens} new in {t2:.1f}s = {tok_s} tok/s")
    print(text[:600])
    gate(f"[{name}] deterministic across two runs", s1 == s2)
    gate(f"[{name}] non-degenerate output", len(set(s2)) > 10, f"{len(set(s2))} distinct tokens")
write_report(f"baseline_B_qwen_{a.bits}bit", rep)
finish("Phase 0 / 05 baseline B (Qwen AR)")
