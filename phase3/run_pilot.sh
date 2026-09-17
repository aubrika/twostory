#!/usr/bin/env bash
# Overnight driver: pilot training, then throughput with the best checkpoint. Everything to logs/phase3/.
cd "$(dirname "$0")/.."; PY=.venv/bin/python; mkdir -p logs/phase3
F='UserWarning|torch_dtype|MatMul8bitLt|Loading checkpoint|use_cache|warning:|^\s*\^|^\s*[0-9]+ \||^\s*\|'
echo "=== pilot start $(date)"
$PY phase3/30_pilot.py "$@" 2>&1 | grep -v -E "$F" > logs/phase3/30_pilot.log; echo "pilot exit ${PIPESTATUS[0]} $(date)"
echo "=== throughput (best ckpt) start $(date)"
$PY phase3/31_throughput.py --ckpt checkpoints/phase3_pilot/best.pt 2>&1 | grep -v -E "$F" > logs/phase3/31_throughput_trained.log; echo "throughput exit ${PIPESTATUS[0]} $(date)"
echo "=== done $(date)"
