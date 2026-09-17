"""Phase 0 / 01: lineage premise. Dream-7B and Qwen2.5-7B must share tokenizer, hidden size, layer count."""
import hashlib, json
from transformers import AutoConfig
from twostory.util import gate, finish, write_report, load_tokenizer, REPO, DREAM_ID, QWEN_ID

rep = {}
dc = AutoConfig.from_pretrained(DREAM_ID, trust_remote_code=True)
qc = AutoConfig.from_pretrained(QWEN_ID)
keys = ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "intermediate_size", "vocab_size", "rope_theta", "rms_norm_eps", "hidden_act", "tie_word_embeddings"]
rep["config"] = {k: {"dream": getattr(dc, k, None), "qwen": getattr(qc, k, None)} for k in keys}
for k in keys:
    gate(f"config.{k} equal", getattr(dc, k, None) == getattr(qc, k, None), f"{getattr(dc,k,None)} vs {getattr(qc,k,None)}")
rep["dream_mask_token_id"] = dc.mask_token_id

dt = load_tokenizer("dream"); qt = load_tokenizer("qwen")
dv, qv = dt.get_vocab(), qt.get_vocab()
extra_in_dream = sorted(set(dv) - set(qv)); extra_in_qwen = sorted(set(qv) - set(dv))
rep["extra_in_dream"] = {t: dv[t] for t in extra_in_dream}
rep["extra_in_qwen"] = {t: qv[t] for t in extra_in_qwen}
shared = set(dv) & set(qv)
mismatch = [t for t in shared if dv[t] != qv[t]]
rep["shared_vocab_size"] = len(shared); rep["id_mismatches"] = len(mismatch)
h = lambda v: hashlib.sha256(json.dumps(sorted(v.items()), ensure_ascii=False).encode()).hexdigest()
rep["shared_vocab_hash_dream"] = h({t: dv[t] for t in shared}); rep["shared_vocab_hash_qwen"] = h({t: qv[t] for t in shared})
print("extra tokens only in Dream:", rep["extra_in_dream"]); print("extra tokens only in Qwen:", rep["extra_in_qwen"])
gate("shared vocab ids identical", not mismatch and rep["shared_vocab_hash_dream"] == rep["shared_vocab_hash_qwen"], f"{len(shared)} shared, {len(mismatch)} mismatched")
gate("Dream extras are only <|beginoftext|>/<|mask|>", set(extra_in_dream) <= {"<|beginoftext|>", "<|mask|>"})
gate("no Qwen-only tokens", not extra_in_qwen)
gate("Dream extras sit inside Qwen embedding table", all(i < qc.vocab_size for i in rep["extra_in_dream"].values()))

samples = [
    "The quick brown fox jumps over the lazy dog.",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
    "Über den Wolken muss die Freiheit wohl grenzenlos sein. 日本語のテキストも含みます。",
    "  leading spaces, trailing spaces   \n\n\ttabs and\r\nnewlines",
    "Numbers 1234567890 3.14159 1e-9 and symbols !@#$%^&*()_+-=[]{}|;':\",./<>?",
    "<|endoftext|><|im_start|>user\nhi<|im_end|>",
    "🙂🚀 emoji and ünïcödé combining: é",
    "A" * 500,
]
enc_mismatch = []
for s in samples:
    a = dt(s, add_special_tokens=False).input_ids; b = qt(s, add_special_tokens=False).input_ids
    if a != b: enc_mismatch.append((s[:40], a[:10], b[:10]))
rep["encoding_mismatches"] = enc_mismatch
gate("encodings identical on sample strings (no special tokens)", not enc_mismatch, f"{len(samples)} strings")
rep["dream_bos"] = (dt.bos_token, dt.bos_token_id); rep["qwen_bos"] = (qt.bos_token, qt.bos_token_id)
rep["eos"] = (dt.eos_token_id, qt.eos_token_id)
print("bos:", rep["dream_bos"], rep["qwen_bos"], "eos:", rep["eos"])
gate("eos ids equal", dt.eos_token_id == qt.eos_token_id)
# Dream prepends <|beginoftext|> when add_special_tokens=True; Qwen adds nothing. Record it - the
# bridge must feed the context tower Qwen-style (no BOS) and the denoiser Dream-style.
rep["dream_add_special"] = dt("hi").input_ids; rep["qwen_add_special"] = qt("hi").input_ids
print("add_special_tokens=True: dream", rep["dream_add_special"], "qwen", rep["qwen_add_special"])
write_report("01_lineage", rep)
finish("Phase 0 / 01 lineage")
