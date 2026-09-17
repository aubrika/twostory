"""Build, hash and freeze the benchmark prompt set (docs/BENCHMARK.md §3).

Tiers: 128/512/2048 tokens x20, 4096 x10, drawn from the held-out split (never trained on), plus
hand-written task prompts (edit HANDWRITTEN below, then rebuild - the hash changes and that is the point).
Long tiers concatenate consecutive held-out sequences (eos-separated docs, same packing as training).
Output: data/prompt_set.json with sha256 in data/prompt_set.sha256. Run once; never regenerate mid-study."""
import argparse, hashlib, json
import numpy as np
from twostory.util import load_tokenizer, REPO, gate, finish

HANDWRITTEN = [
    # (name, text). Base models: plain continuation style, no chat template.
    ("instr_summary", "Summarize the following in one paragraph.\n\nThe committee met on three occasions to review the proposal for a new footbridge across the river. Concerns were raised about cost, flood risk, and the impact on the heron colony downstream. A revised design with a single span was accepted at the third meeting.\n\nSummary:"),
    ("code_python", "# Python 3\n# Write a function that returns the n-th Fibonacci number using memoization.\ndef fib(n, memo=None):\n"),
    ("code_sql", "-- Table: orders(id, customer_id, total, created_at)\n-- Query: total revenue per customer for 2024, highest first.\nSELECT"),
    ("qa_factual", "Q: What is the boiling point of water at sea level in degrees Celsius?\nA: 100\n\nQ: How many sides does a hexagon have?\nA:"),
    ("qa_reasoning", "Q: A train leaves at 9:15 and the journey takes 2 hours 50 minutes. When does it arrive?\nA: Let's think step by step."),
    ("list_gen", "Ten common birds found in European gardens:\n1."),
    ("dialogue", "Customer: Hi, I ordered a kettle last week and it arrived with a cracked lid.\nAgent:"),
    ("math_word", "Q: A bakery sells muffins at 3 for $5. How much do 12 muffins cost?\nA:"),
    ("story_open", "The last train had gone, and the station was empty except for a man in a grey coat who"),
    ("explain", "Explain, in simple terms, why the sky is blue.\n\n"),
]
TIERS = {"short": (128, 20), "medium": (512, 20), "long": (2048, 20), "vlong": (4096, 10)}
GEN_TOKENS = 256

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20260917); a = ap.parse_args()
tok = load_tokenizer("qwen")
meta = json.load(open(REPO / "data/tokens/meta.json"))
L = meta["seq_len"]
held = np.memmap(REPO / "data/tokens/heldout.bin", dtype=np.uint32, mode="r").reshape(-1, L)
rng = np.random.default_rng(a.seed)
prompts = []
for tier, (n_tok, count) in TIERS.items():
    n_seq = max(1, -(-n_tok // L))            # ceil
    starts = rng.choice(len(held) - n_seq, size=count, replace=False)
    for k, s in enumerate(sorted(starts.tolist())):
        ids = held[s:s + n_seq].reshape(-1)[:n_tok].astype(int).tolist()
        prompts.append({"id": f"{tier}_{k:02d}", "tier": tier, "source": "heldout", "heldout_seq": s,
                        "prompt_tokens": n_tok, "gen_tokens": GEN_TOKENS, "ids": ids})
for name, text in HANDWRITTEN:
    ids = tok(text, add_special_tokens=False).input_ids
    prompts.append({"id": f"hand_{name}", "tier": "hand", "source": "handwritten", "text": text,
                    "prompt_tokens": len(ids), "gen_tokens": GEN_TOKENS, "ids": ids})
for p in prompts:
    p["text_preview"] = tok.decode(p["ids"][:40])[:160]
blob = json.dumps({"seed": a.seed, "tiers": TIERS, "gen_tokens": GEN_TOKENS, "heldout_sha256": meta["sha256_heldout"], "prompts": prompts}, indent=1)
h = hashlib.sha256(blob.encode()).hexdigest()
(REPO / "data/prompt_set.json").write_text(blob); (REPO / "data/prompt_set.sha256").write_text(h + "\n")
print(f"{len(prompts)} prompts, sha256 {h}")
for tier in list(TIERS) + ["hand"]:
    print(f"  {tier}: {sum(p['tier']==tier for p in prompts)}")
gate("prompt set written and hashed", (REPO / "data/prompt_set.json").exists() and len(h) == 64)
gate("no mask/bos ids in prompts", all(151665 not in p["ids"] and 151666 not in p["ids"] for p in prompts))
finish("benchmark prompt set")
