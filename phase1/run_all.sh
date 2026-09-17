#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."; PY=.venv/bin/python; mkdir -p logs/phase1
for s in phase1/1*.py; do n=$(basename "$s" .py); echo; echo "################ $n"
  $PY "$s" "$@" 2>&1 | grep -v -E "UserWarning|torch_dtype|MatMul8bitLt|Loading checkpoint" | tee "logs/phase1/$n.log"
  grep -q ': PASS  ([0-9]*/[0-9]* gates passed)' "logs/phase1/$n.log" || { echo; echo "STOP: $n did not PASS."; exit 1; }
done; echo; echo "Phase 1: ALL PASS"
