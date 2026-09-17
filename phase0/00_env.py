"""Phase 0 / 00: environment. torch XPU, bitsandbytes int8 on XPU, SDPA on XPU, disk/RAM report."""
import shutil
import torch
from twostory.util import gate, finish, write_report, versions, gib, _meminfo, REPO

rep = {"versions": versions()}
print(rep["versions"])

gate("torch.xpu available", torch.xpu.is_available())
mi = _meminfo()
rep["ram_total_gib"] = gib(mi["MemTotal"]); rep["ram_avail_gib"] = gib(mi["MemAvailable"])
rep["swap_total_gib"] = gib(mi["SwapTotal"]); rep["swap_free_gib"] = gib(mi["SwapFree"])
rep["disk_free_gib"] = gib(shutil.disk_usage(REPO).free)
print({k: v for k, v in rep.items() if k != "versions"})
gate("disk free >= 40 GiB", rep["disk_free_gib"] >= 40, f"{rep['disk_free_gib']} GiB")

# bf16 matmul + SDPA on XPU
x = torch.randn(2, 4, 64, 128, device="xpu", dtype=torch.bfloat16)
mask = torch.ones(64, 64, device="xpu", dtype=torch.bool).tril()
y = torch.nn.functional.scaled_dot_product_attention(x, x, x, attn_mask=mask)
gate("SDPA bf16 with bool mask on XPU", torch.isfinite(y.float()).all().item())

# bitsandbytes int8 Linear on XPU (the thing the whole memory plan rests on)
try:
    import bitsandbytes as bnb
    lin = torch.nn.Linear(256, 512, bias=False).to(torch.bfloat16)
    q = bnb.nn.Linear8bitLt(256, 512, bias=False, has_fp16_weights=False)
    q.weight = bnb.nn.Int8Params(lin.weight.data.clone(), requires_grad=False, has_fp16_weights=False)
    q = q.to("xpu")
    inp = torch.randn(3, 256, device="xpu", dtype=torch.bfloat16)
    ref = lin.to("xpu")(inp).float()
    out = q(inp).float()
    err = ((out - ref).norm() / ref.norm()).item()
    rep["bnb_int8_rel_err"] = err
    gate("bitsandbytes Linear8bitLt forward on XPU", torch.isfinite(out).all().item() and err < 0.1, f"rel err {err:.4f}")
    gate("int8 weight dtype", q.weight.dtype == torch.int8, str(q.weight.dtype))
except Exception as e:  # noqa
    import traceback; traceback.print_exc()
    gate("bitsandbytes Linear8bitLt forward on XPU", False, repr(e))

write_report("00_env", rep)
finish("Phase 0 / 00 env")
