#!/usr/bin/env bash
# Wait (by systemd unit name, never by PID) for the benchmark unit to finish, then run the distillation probe alone on the GPU.
cd "$(dirname "$0")/.."
while [ "$(systemctl --user is-active twostory-bench.service 2>/dev/null)" = "active" ]; do sleep 60; done
echo "[probe $(date '+%F %T')] bench unit inactive; starting probe"
for attempt in 1 2 3; do
  .venv/bin/python -u -m bench.distill_probe >> logs/phase5/probe.log 2>&1; rc=$?
  echo "[probe $(date '+%F %T')] attempt $attempt exit rc=$rc"
  if [ $rc -eq 0 ] || [ $rc -eq 4 ]; then break; fi; sleep 30
done
