"""Phase 0 / 06: peft LoRA forward+backward on XPU, on (a) a bf16 tiny model and (b) an 8-bit bnb tiny model
with gradient checkpointing - the exact configuration Phase 2+ trains in. Also checks SDPA with the
bridge's mask shape (block bidirectional + causal over context)."""
import torch, torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from twostory.util import gate, finish, write_report, bnb_config, REPO, DEVICE

torch.manual_seed(0)
rep = {}
cfg = Qwen2Config(hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=8,
                  num_key_value_heads=2, vocab_size=1000, max_position_embeddings=512, tie_word_embeddings=False)
targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lcfg = LoraConfig(r=8, lora_alpha=16, target_modules=targets, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")

def train_step(model, tag):
    ids = torch.randint(0, 1000, (2, 64), device=DEVICE)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    tr = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_tr = sum(p.numel() for _, p in tr); n_all = sum(p.numel() for p in model.parameters())
    bad = [n for n, p in tr if p.grad is None or not torch.isfinite(p.grad).all()]
    frozen_with_grad = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    print(f"[{tag}] loss {out.loss.item():.4f} trainable {n_tr}/{n_all} lora-only={all('lora' in n for n,_ in tr)}")
    gate(f"[{tag}] loss finite", torch.isfinite(out.loss).item())
    gate(f"[{tag}] all trainable params got finite grads", not bad, f"{len(bad)} bad: {bad[:3]}")
    gate(f"[{tag}] only LoRA params trainable", all("lora" in n for n, _ in tr))
    gate(f"[{tag}] frozen params have no grad", not frozen_with_grad)
    rep[tag] = {"loss": out.loss.item(), "trainable": n_tr, "total": n_all}

# (a) bf16 tiny model
m = Qwen2ForCausalLM(cfg).to(DEVICE, torch.bfloat16)
pm = get_peft_model(m, lcfg)
train_step(pm, "bf16+lora")

# (b) 8-bit bnb tiny model + LoRA + grad checkpointing (the real training config)
tiny = REPO / "logs" / "phase0" / "tiny_qwen2"
Qwen2ForCausalLM(cfg).save_pretrained(tiny)
q = AutoModelForCausalLM.from_pretrained(tiny, quantization_config=bnb_config(8), device_map={"": DEVICE}, torch_dtype=torch.bfloat16)
print("8bit q_proj dtype:", q.model.layers[0].self_attn.q_proj.weight.dtype)
gate("[int8+lora] base weights are int8", q.model.layers[0].self_attn.q_proj.weight.dtype == torch.int8)
q = prepare_model_for_kbit_training(q, use_gradient_checkpointing=True)
pq = get_peft_model(q, lcfg)
pq.train()
train_step(pq, "int8+lora+ckpt")

# (c) SDPA with the bridge mask: queries = block of B tokens, keys = context of C tokens + the block.
B, C, H, D = 32, 96, 4, 64
qq = torch.randn(1, H, B, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
kk = torch.randn(1, H, C + B, D, device=DEVICE, dtype=torch.bfloat16)
vv = torch.randn(1, H, C + B, D, device=DEVICE, dtype=torch.bfloat16)
mask = torch.ones(B, C + B, device=DEVICE, dtype=torch.bool)   # block sees all context + itself (bidirectional)
o = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask)
ref = torch.softmax((qq.float() @ kk.float().transpose(-1, -2)) / D**0.5, -1) @ vv.float()
err = ((o.float() - ref).norm() / ref.norm()).item()
o.sum().backward()
gate("[sdpa] bool-mask SDPA matches reference on XPU", err < 2e-2, f"rel err {err:.4f}")
gate("[sdpa] backward through SDPA on XPU", qq.grad is not None and torch.isfinite(qq.grad.float()).all().item())
rep["sdpa_rel_err"] = err
write_report("06_lora_xpu", rep)
finish("Phase 0 / 06 LoRA on XPU")
