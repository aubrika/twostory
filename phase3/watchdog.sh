#!/usr/bin/env bash
# Progress-based watchdog. Runs the pilot; if its per-step log stops advancing for STALE_SEC (driver wedge), kills it
# and relaunches with --resume from the last checkpoint. Logs are raw (no grep buffering). Then throughput.
cd "$(dirname "$0")/.."; PY=.venv/bin/python; RUN=phase3_pilot; STALE_SEC=${STALE_SEC:-600}; MAX_RESTARTS=${MAX_RESTARTS:-6}
mkdir -p logs/phase3; PROGRESS=checkpoints/$RUN/train.jsonl; restarts=0
log(){ echo "[watchdog $(date '+%F %T')] $*"; }
while :; do
  log "launching pilot (restart #$restarts)"
  $PY phase3/30_pilot.py --run-name $RUN --resume "$@" >> logs/phase3/30_pilot.log 2>&1 &
  pid=$!; last_size=-1; last_change=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    size=$(stat -c %s "$PROGRESS" 2>/dev/null || echo 0)
    if [ "$size" != "$last_size" ]; then last_size=$size; last_change=$(date +%s); fi
    if [ $(( $(date +%s) - last_change )) -gt $STALE_SEC ]; then
      log "STALE: no progress for ${STALE_SEC}s -> kill -9 $pid"; kill -9 $pid; sleep 10
      journalctl -k --since "-${STALE_SEC}s" --no-pager 2>/dev/null | grep -E "Timedout|Engine reset" | tail -2
      break
    fi
  done
  wait $pid; rc=$?
  if [ $rc -eq 0 ]; then log "pilot finished OK"; break; fi
  restarts=$((restarts+1)); log "pilot exited rc=$rc"
  if [ $restarts -ge $MAX_RESTARTS ]; then log "giving up after $restarts restarts"; exit 1; fi
  sleep 20
done
log "throughput with best checkpoint"
$PY phase3/31_throughput.py --ckpt checkpoints/$RUN/best.pt >> logs/phase3/31_throughput_trained.log 2>&1; log "throughput rc=$?"
log "done"
