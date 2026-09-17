"""Phase 0 / 07: pre-tokenize FineWeb-Edu to disk once. Packs docs (eos-separated) into fixed 1024-token
sequences as uint32; every `heldout_every`-th sequence goes to heldout.bin (2%), never to be trained on."""
import argparse, json, os, time
import numpy as np
import pyarrow.parquet as pq
from twostory.util import gate, finish, write_report, load_tokenizer, sha256_file, REPO

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
ap = argparse.ArgumentParser()
ap.add_argument("--parquet-dir", default=str(REPO / "data" / "fineweb-edu" / "sample" / "10BT"))
ap.add_argument("--out", default=str(REPO / "data" / "tokens"))
ap.add_argument("--target-tokens", type=int, default=400_000_000)
ap.add_argument("--seq-len", type=int, default=1024)
ap.add_argument("--heldout-every", type=int, default=50)
ap.add_argument("--batch-docs", type=int, default=1024)
a = ap.parse_args()

tok = load_tokenizer("qwen")
eos = tok.eos_token_id
files = sorted(p for p in os.listdir(a.parquet_dir) if p.endswith(".parquet"))
gate("parquet shards present", len(files) > 0, f"{files}")
out = REPO / a.out if not os.path.isabs(a.out) else a.out
os.makedirs(out, exist_ok=True)
train_f = open(os.path.join(out, "train.bin"), "wb"); held_f = open(os.path.join(out, "heldout.bin"), "wb")

carry = np.zeros(0, dtype=np.uint32); n_seq = n_held = n_docs = 0; n_tok = 0; max_id = 0; t0 = time.time()
done = False
for fn in files:
    pf = pq.ParquetFile(os.path.join(a.parquet_dir, fn))
    for batch in pf.iter_batches(batch_size=a.batch_docs, columns=["text"]):
        texts = batch.column("text").to_pylist()
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        n_docs += len(texts)
        chunks = [carry] + [np.asarray(e + [eos], dtype=np.uint32) for e in enc]
        flat = np.concatenate(chunks)
        n_full = len(flat) // a.seq_len
        seqs = flat[: n_full * a.seq_len].reshape(n_full, a.seq_len)
        carry = flat[n_full * a.seq_len:]
        if n_full:
            max_id = max(max_id, int(seqs.max()))
            idx = np.arange(n_seq, n_seq + n_full)
            held_mask = (idx % a.heldout_every) == 0
            held_f.write(seqs[held_mask].tobytes()); train_f.write(seqs[~held_mask].tobytes())
            n_held += int(held_mask.sum()); n_seq += n_full; n_tok += n_full * a.seq_len
        if n_tok // 20_000_000 != (n_tok - n_full * a.seq_len) // 20_000_000:
            print(f"  {n_tok/1e6:.0f}M tokens, {n_docs} docs, {n_seq} seqs, {(time.time()-t0):.0f}s", flush=True)
        if n_tok >= a.target_tokens:
            done = True; break
    if done: break
train_f.close(); held_f.close()

meta = {"tokenizer": "Qwen/Qwen2.5-7B", "eos_id": eos, "seq_len": a.seq_len, "dtype": "uint32",
        "n_sequences": n_seq, "n_heldout": n_held, "n_train": n_seq - n_held, "n_tokens": n_tok, "n_docs": n_docs,
        "heldout_every": a.heldout_every, "max_token_id": max_id, "source_files": files, "seconds": round(time.time() - t0),
        "sha256_heldout": sha256_file(os.path.join(out, "heldout.bin")), "sha256_train": sha256_file(os.path.join(out, "train.bin"))}
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
print(json.dumps(meta, indent=2))
gate("reached target token count", n_tok >= a.target_tokens, f"{n_tok/1e6:.1f}M")
gate("held-out split written and hashed", n_held > 0 and len(meta["sha256_heldout"]) == 64, f"{n_held} seqs, sha256 {meta['sha256_heldout'][:16]}...")
gate("all token ids < vocab_size", max_id < 152064, str(max_id))
gate("heldout fraction ~2%", abs(n_held / n_seq - 1 / a.heldout_every) < 0.002)
write_report("07_pretokenize", meta)
finish("Phase 0 / 07 pretokenize")
