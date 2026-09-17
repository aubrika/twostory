"""Phase 0 / 04: Baseline A. Dream-7B's own diffusion generation, deterministic, with tokens/sec.
The saved token ids are the Phase 1 regression target: the zero-init bridged model must reproduce them exactly."""
import argparse, time, torch
from twostory.util import gate, finish, write_report, MemTracker, load_dream, load_tokenizer, DENOISER_BITS
from twostory.prompts import PROMPT_SHORT, build_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--bits", type=int, default=DENOISER_BITS)
ap.add_argument("--new-tokens", type=int, default=128)
ap.add_argument("--steps", type=int, default=128)
ap.add_argument("--prompts", default="short,p512")
a = ap.parse_args()

tok = load_tokenizer("dream")
with MemTracker() as mt:
    model = load_dream(a.bits)
print("load:", mt.summary())

def gen(prompt: str, new_tokens: int, steps: int):
    ids = tok(prompt, return_tensors="pt").input_ids.to("xpu")   # Dream style: prepends <|beginoftext|>
    torch.xpu.synchronize(); t0 = time.time()
    out = model.diffusion_generate(ids, max_new_tokens=new_tokens, steps=steps, temperature=0.0,
                                   alg="entropy", alg_temp=None, return_dict_in_generate=True)
    torch.xpu.synchronize(); dt = time.time() - t0
    seq = out.sequences[0, ids.shape[1]:].tolist()
    return ids.shape[1], seq, dt

print("warmup..."); gen(PROMPT_SHORT, 16, 16)
rep = {"bits": a.bits, "new_tokens": a.new_tokens, "steps": a.steps, "alg": "entropy", "temperature": 0.0, "prompts": {}}
for name in a.prompts.split(","):
    prompt = PROMPT_SHORT if name == "short" else build_prompt(tok, int(name[1:]) if name[1:].isdigit() else {"p2k": 2048}[name])
    plen, s1, t1 = gen(prompt, a.new_tokens, a.steps)
    with MemTracker() as gm:
        _, s2, t2 = gen(prompt, a.new_tokens, a.steps)
    text = tok.decode(s2)
    tok_s = round(a.new_tokens / t2, 2)
    rep["prompts"][name] = {"prompt_tokens": plen, "ids": s2, "text": text, "seconds": round(t2, 2), "tok_s": tok_s,
                           "deterministic": s1 == s2, "gen_mem": gm.summary()}
    print(f"\n=== [{name}] prompt {plen} tok -> {a.new_tokens} new in {t2:.1f}s = {tok_s} tok/s (peak VRAM {gm.summary()['vram_peak_alloc_gib']} GiB)")
    print(text[:600])
    gate(f"[{name}] deterministic across two runs", s1 == s2)
    gate(f"[{name}] no mask tokens left", model.config.mask_token_id not in s2)
    gate(f"[{name}] non-degenerate output", len(set(s2)) > 10, f"{len(set(s2))} distinct tokens")
write_report(f"baseline_A_dream_{a.bits}bit", rep)
finish("Phase 0 / 04 baseline A (Dream)")
