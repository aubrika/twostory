#!/usr/bin/env bash
# Run every Phase 0 check in order; stop at the first FAIL. Logs land in logs/phase0/NN.log.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs/phase0
for s in phase0/0*.py; do
  n=$(basename "$s" .py)
  echo; echo "################ $n"
  if ! $PY "$s" "$@" 2>&1 | tee "logs/phase0/$n.log"; then :; fi
  if ! grep -q ': PASS  ([0-9]*/[0-9]* gates passed)' "logs/phase0/$n.log"; then
    echo; echo "STOP: $n did not PASS. Read logs/phase0/$n.log and decide."; exit 1
  fi
done
echo; echo "Phase 0: ALL PASS"
