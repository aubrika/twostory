"""Tiny-model reproduction of (a) eval-mode NaN and (b) the >5s GPU job at n=23 blocks, to isolate the kernel path."""
import time, math, torch, torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM
from twostory.dream import DreamConfig, DreamModel
from twostory.model import TwoStory
from twostory.bridge import BridgeConfig
torch.manual_seed(0)
H, NH, NKV, NL = 256, 8, 2, 4
dc = DreamConfig(hidden_size=H, intermediate_size=512, num_hidden_layers=NL, num_attention_heads=NH, num_key_value_heads=NKV,
                 vocab_size=2000, mask_token_id=1999, eos_token_id=0, pad_token_id=0, max_position_embeddings=4096, tie_word_embeddings=False)
qc = Qwen2Config(hidden_size=H, intermediate_size=512, num_hidden_layers=NL, num_attention_heads=NH, num_key_value_heads=NKV,
                 vocab_size=2000, max_position_embeddings=4096, tie_word_embeddings=False)
dream = DreamModel(dc).to("xpu", torch.bfloat16); qwen = Qwen2ForCausalLM(qc).to("xpu", torch.bfloat16)
for p in list(dream.parameters()) + list(qwen.parameters()): p.requires_grad_(False)
ts = TwoStory(dream, qwen, BridgeConfig(layers=list(range(NL))), lora=None)
with torch.no_grad():
    for b in ts.bridges: b.out_proj.weight.normal_(0, 0.02)   # non-zero so bridge output matters
L = 1024
def run(ctx_len, grad, tag):
    seq = torch.randint(1, 1998, (1, L), device="xpu")
    n = (L - ctx_len) // 32
    torch.xpu.synchronize(); t0 = time.perf_counter()
    if grad:
        ts.train(); loss, info = ts.training_loss(seq, ctx_len, 32)
    else:
        ts.eval()
        with torch.no_grad(): loss, info = ts.training_loss(seq, ctx_len, 32)
    torch.xpu.synchronize(); dt = time.perf_counter() - t0
    print(f"  {tag} ctx {ctx_len:4d} n={n:2d} grad={grad}: loss {loss.item():.4f} {'OK ' if math.isfinite(loss.item()) else 'NaN'} {dt*1e3:6.0f} ms", flush=True)
print("--- sweep ctx_len (n blocks) in eval/no_grad and train modes")
for ctx in (768, 512, 384, 320, 288, 256):
    run(ctx, False, "eval "); run(ctx, True, "train")
print("--- isolate SDPA with expanded K/V + expanded bool mask vs contiguous vs math reference (no_grad)")
n, B1, HD = 23, 33, H // NH
q = torch.randn(n, NH, B1, HD, device="xpu", dtype=torch.bfloat16)
k1 = torch.randn(1, NH, L, HD, device="xpu", dtype=torch.bfloat16); v1 = torch.randn(1, NH, L, HD, device="xpu", dtype=torch.bfloat16)
starts = 288 + 32 * torch.arange(n, device="xpu")
mask_exp = (torch.arange(L, device="xpu")[None, None, None, :] < starts[:, None, None, None]).expand(n, 1, B1, L)
ref = torch.softmax((q.float() @ k1.float().expand(n,-1,-1,-1).transpose(-1,-2)) / math.sqrt(HD) + torch.where(mask_exp, 0.0, float("-inf")).float(), -1) @ v1.float().expand(n,-1,-1,-1)
with torch.no_grad():
    for name, kk, vv, mm in [("expanded kv + expanded mask", k1.expand(n,-1,-1,-1), v1.expand(n,-1,-1,-1), mask_exp),
                             ("contiguous kv + expanded mask", k1.expand(n,-1,-1,-1).contiguous(), v1.expand(n,-1,-1,-1).contiguous(), mask_exp),
                             ("expanded kv + contiguous mask", k1.expand(n,-1,-1,-1), v1.expand(n,-1,-1,-1), mask_exp.contiguous()),
                             ("contiguous both", k1.expand(n,-1,-1,-1).contiguous(), v1.expand(n,-1,-1,-1).contiguous(), mask_exp.contiguous()),
                             ("expanded kv + float additive mask", k1.expand(n,-1,-1,-1), v1.expand(n,-1,-1,-1), torch.where(mask_exp, 0.0, float("-inf")).to(torch.bfloat16))]:
        torch.xpu.synchronize(); t0 = time.perf_counter()
        o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mm)
        torch.xpu.synchronize(); dt = time.perf_counter() - t0
        err = ((o.float() - ref).norm() / ref.norm()).item()
        print(f"  {name:36s}: finite={torch.isfinite(o).all().item()} rel_err={err:.4f} {dt*1e3:.1f} ms", flush=True)
print("DIAG SDPA COMPLETE")
