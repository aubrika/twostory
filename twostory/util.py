"""Shared helpers: gates, memory tracking, model loading, reports."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import warnings
import torch

warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs"
DREAM_ID = "Dream-org/Dream-v0-Base-7B"
QWEN_ID = "Qwen/Qwen2.5-7B"
# Pinned Hub revisions (Phase 0, 2026-09-17). Baseline A was recorded against these.
DREAM_REV = "6572adb5535263e4d1a337b56942ba48b6dee2a9"
QWEN_REV = "d149729398750b98c0af14eb82c78cfe92750796"
DEVICE = "xpu:0"
# Memory plan (revised 2026-09-17 after measuring bnb int8 at 4-8x slower than bf16 on XPU):
# denoiser runs every denoising step -> bf16; context tower runs once per committed block -> int8.
DENOISER_BITS = 16
CONTEXT_BITS = 8


# --------------------------------------------------------------------------- gates
_gates: list[tuple[str, bool]] = []


def gate(name: str, ok: bool, detail: str = "") -> bool:
    _gates.append((name, bool(ok)))
    print(f"[GATE] {name}: {'PASS' if ok else 'FAIL'}" + (f"  ({detail})" if detail else ""), flush=True)
    return bool(ok)


def finish(phase: str) -> None:
    """Print the overall verdict and exit non-zero if any gate failed."""
    ok = all(g for _, g in _gates)
    print("=" * 70)
    print(f"{phase}: {'PASS' if ok else 'FAIL'}  ({sum(g for _, g in _gates)}/{len(_gates)} gates passed)")
    print("=" * 70, flush=True)
    sys.exit(0 if ok else 1)


def write_report(name: str, data: dict, subdir: str = "phase0") -> Path:
    d = LOGS / subdir
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(json.dumps(data, indent=2, default=str))
    print(f"[report] {p}")
    return p


# --------------------------------------------------------------------------- memory
def _meminfo() -> dict[str, int]:
    out = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        out[k] = int(v.strip().split()[0]) * 1024  # bytes
    return out


def gib(n: float) -> float:
    return round(n / 2**30, 2)


class MemTracker:
    """Peak VRAM (torch.xpu), peak RSS (ru_maxrss), and swap growth over a region."""

    def __enter__(self):
        torch.xpu.synchronize()
        torch.xpu.reset_peak_memory_stats()
        mi = _meminfo()
        self.swap_used_start = mi["SwapTotal"] - mi["SwapFree"]
        self.rss_peak_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        torch.xpu.synchronize()
        mi = _meminfo()
        self.seconds = round(time.time() - self.t0, 1)
        self.vram_peak_alloc = torch.xpu.max_memory_allocated()
        self.vram_peak_reserved = torch.xpu.max_memory_reserved()
        self.vram_now = torch.xpu.memory_allocated()
        self.rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.swap_growth = (mi["SwapTotal"] - mi["SwapFree"]) - self.swap_used_start

    def summary(self) -> dict:
        return {
            "seconds": self.seconds,
            "vram_peak_alloc_gib": gib(self.vram_peak_alloc),
            "vram_peak_reserved_gib": gib(self.vram_peak_reserved),
            "vram_now_gib": gib(self.vram_now),
            "rss_peak_gib": gib(self.rss_peak),
            "swap_growth_gib": gib(self.swap_growth),
        }


# --------------------------------------------------------------------------- loading
def bnb_config(bits: int):
    from transformers import BitsAndBytesConfig

    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    return None


def _load_kwargs(bits: int) -> dict:
    kw = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": DEVICE},
    )
    q = bnb_config(bits)
    if q is not None:
        kw["quantization_config"] = q
    return kw


def load_qwen(bits: int = CONTEXT_BITS):
    """Frozen context tower: Qwen2.5-7B causal LM."""
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(QWEN_ID, revision=QWEN_REV, **_load_kwargs(bits))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_dream(bits: int = DENOISER_BITS, model_id: str = DREAM_ID):
    """Denoiser tower: Dream-7B via its own remote modeling code (unmodified)."""
    from transformers import AutoModel

    m = AutoModel.from_pretrained(model_id, revision=DREAM_REV if model_id == DREAM_ID else None, trust_remote_code=True, **_load_kwargs(bits))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_tokenizer(which: str = "qwen"):
    from transformers import AutoTokenizer

    if which == "qwen":
        return AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REV)
    return AutoTokenizer.from_pretrained(DREAM_ID, revision=DREAM_REV, trust_remote_code=True)


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def versions() -> dict:
    import transformers, peft, bitsandbytes, accelerate

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "accelerate": accelerate.__version__,
        "xpu_device": torch.xpu.get_device_name(0) if torch.xpu.is_available() else None,
    }


# --------------------------------------------------------------------------- provenance
def provenance() -> dict:
    """Commit hash, dirty flag, package versions, and pip freeze. Attach to every results file."""
    import subprocess

    def sh(cmd):
        try:
            return subprocess.check_output(cmd, cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "commit": sh(["git", "rev-parse", "HEAD"]),
        "dirty": bool(sh(["git", "status", "--porcelain", "--", ":!logs", ":!data"])),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "versions": versions(),
        "dream_rev": DREAM_REV,
        "qwen_rev": QWEN_REV,
        "pip_freeze": sh([sys.executable, "-m", "pip", "freeze"]) or sh(["uv", "pip", "freeze", "--python", sys.executable]),
    }


def gpu_telemetry() -> dict:
    """B70 clock (MHz), throttle flags, labelled temperatures (C), energy counters (uJ) from the xe driver's sysfs."""
    base = Path("/sys/class/drm/card0/device")

    def rd(p):
        try:
            return open(p).read().strip()
        except Exception:
            return None

    def num(p):
        v = rd(p)
        return int(v) if v is not None and v.lstrip("-").isdigit() else None

    fq = base / "tile0/gt0/freq0"
    out = {"act_freq_mhz": num(fq / "act_freq"), "cur_freq_mhz": num(fq / "cur_freq"), "max_freq_mhz": num(fq / "max_freq")}
    out["throttle"] = {f.name: num(f) for f in sorted((fq / "throttle").glob("*")) if f.is_file()} if (fq / "throttle").exists() else {}
    temps, energy = {}, {}
    for hw in sorted((base / "hwmon").glob("hwmon*")) if (base / "hwmon").exists() else []:
        for f in sorted(hw.glob("temp*_input")):
            label = rd(f.with_name(f.name.replace("_input", "_label"))) or f.name
            v = num(f)
            temps[label] = v / 1000 if v is not None else None
        for f in sorted(hw.glob("energy*_input")):
            label = rd(f.with_name(f.name.replace("_input", "_label"))) or f.name
            energy[label] = num(f)
    out["temps_c"] = temps
    out["temp_max_c"] = max((v for v in temps.values() if v is not None), default=None)
    out["energy_uj"] = energy
    return out


def gpu_other_users() -> list[str]:
    """Other processes holding the render node open (benchmark isolation check)."""
    import subprocess

    try:
        txt = subprocess.check_output(["lsof", "-t", "/dev/dri/renderD128"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    pids = sorted({int(p) for p in txt.split() if p and int(p) != os.getpid()})
    names = []
    for pid in pids:
        try:
            names.append(f"{pid}:{open(f'/proc/{pid}/comm').read().strip()}")
        except Exception:
            names.append(str(pid))
    return names
