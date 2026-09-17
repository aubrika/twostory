"""Phase 1 / 11: gradients flow to bridge + LoRA only; towers frozen. One real training step (block-mode
masked-diffusion loss on a held-out sequence) with gradient checkpointing, plus a block-mode generation smoke
test (T0: untrained bridge -> unconditional output expected, but the loop must run and time itself)."""
import json, numpy as np, torch
from twostory.util import gate, finish, write_report, MemTracker, load_tokenizer, REPO, provenance, gib
from twostory.model import TwoStory, LoraSpec
from twostory.bridge import BridgeConfig
from twostory.prompts import PROMPT_SHORT

with MemTracker() as mt:
    ts = TwoStory.load(BridgeConfig(), LoraSpec(rank=16))
print("load:", mt.summary())
pr = ts.param_report(); print("params:", pr)
rep = {"provenance": provenance(), "params": pr}
gate("trainable == bridge + lora", pr["trainable_total"] == pr["bridge"] + pr["lora"])
gate("trainable count in expected range (50M..600M)", 50e6 <= pr["trainable_total"] <= 600e6, f"{pr['trainable_total']/1e6:.1f}M")
gate("context tower fully frozen", all(not p.requires_grad for p in ts.ctx.qwen.parameters()))

# one real training step
meta = json.load(open(REPO / "data/tokens/meta.json"))
held = np.memmap(REPO / "data/tokens/heldout.bin", dtype=np.uint32, mode="r").reshape(-1, meta["seq_len"])
seq = torch.from_numpy(held[0].astype(np.int64)).unsqueeze(0).to("xpu")
ts.dream.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
ts.train()
with MemTracker() as tm:
    loss, info = ts.training_loss(seq, ctx_len=512, block_size=32)
    loss.backward()
print(f"loss {loss.item():.4f} {info}"); print("train step:", tm.summary())
tr = ts.trainable_parameters()
no_grad = [i for i, p in enumerate(tr) if p.grad is None]
nonfinite = [i for i, p in enumerate(tr) if p.grad is not None and not torch.isfinite(p.grad).all()]
frozen_grad = [n for n, p in list(ts.dream.named_parameters()) + list(ts.ctx.qwen.named_parameters()) if not p.requires_grad and p.grad is not None]
# with a zero out_proj, out_proj gets grad but q_lora / norm upstream of it get exactly zero grad at step 0; that's expected
zero_grad_bridge = sum(1 for p in tr[: len(tr) - len(ts.lora_params)] if p.grad is not None and p.grad.abs().sum() == 0)
out_proj_grads = [br.out_proj.weight.grad.abs().mean().item() for br in ts.bridges]
gate("loss finite", torch.isfinite(loss).item(), f"{loss.item():.4f}")
gate("every trainable param received a grad", not no_grad, f"{len(no_grad)} missing")
gate("all grads finite", not nonfinite)
gate("no frozen param has a grad", not frozen_grad, str(frozen_grad[:3]))
gate("bridge out_proj grads are non-zero (signal reaches the bridge)", all(g > 0 for g in out_proj_grads), f"mean |g| layer0 {out_proj_grads[0]:.2e}")
gate("train-step peak VRAM <= 28 GiB", tm.summary()["vram_peak_alloc_gib"] <= 28, f"{tm.summary()['vram_peak_alloc_gib']} GiB")
rep.update({"loss": loss.item(), "info": info, "train_step_mem": tm.summary(), "zero_grad_bridge_params_at_step0": zero_grad_bridge,
            "out_proj_grad_mean_abs": out_proj_grads})
ts.dream.zero_grad(set_to_none=True)

# block-mode smoke test (T0)
ts.eval(); ts.dream.gradient_checkpointing_disable()
tokq = load_tokenizer("qwen")
ids = tokq(PROMPT_SHORT, return_tensors="pt").input_ids.to("xpu")
ts.generate_block(ids, max_new_tokens=32, block_size=32, steps=8)  # warmup
with MemTracker() as gm:
    r = ts.generate_block(ids, max_new_tokens=128, block_size=32, steps=8)
txt = tokq.decode(r["ids"])
print(f"T0 block mode: {len(r['ids'])} tok in {r['total_seconds']:.2f}s (ttft {r['ttft_seconds']:.2f}s, prefill {r['prefill_seconds']:.3f}s) = {len(r['ids'])/r['total_seconds']:.1f} tok/s; per-block {[round(b['seconds'],2) for b in r['blocks']]}")
print("T0 text (unconditional expected):", repr(txt[:300]))
gate("block-mode loop runs to length", len(r["ids"]) == 128 or (r["ids"] and r["ids"][-1] == ts.eos_id), str(len(r["ids"])))
gate("block-mode output has no mask tokens", ts.mask_id not in r["ids"])
gate("cache extended block by block", ts.ctx.length == ids.shape[1] + len(r["ids"]) - (0 if len(r["ids"]) < 128 else 32), f"cache {ts.ctx.length}")
rep["t0_block_mode"] = {k: v for k, v in r.items() if k != "ids"} | {"text": txt, "gen_mem": gm.summary()}
write_report("11_trainable", rep, subdir="phase1")
finish("Phase 1 / 11 trainable + block mode")
