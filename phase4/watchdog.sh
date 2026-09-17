#!/usr/bin/env bash
# Runs ONE segment of phase4/40_run.py. Progress watchdog: if the step log stalls (xe driver wedge) -> kill -9 and resume
# from the last checkpoint. A clean exit (segment limit, STOP, plan complete, early stop) ends the watchdog too.
cd "$(dirname "$0")/.."; PY=.venv/bin/python; RUN=${RUN:-phase4_run}; STALE_SEC=${STALE_SEC:-600}; MAX_RESTARTS=${MAX_RESTARTS:-8}
mkdir -p logs/phase4; PROGRESS=checkpoints/$RUN/train.jsonl; restarts=0
log(){ echo "[watchdog $(date '+%F %T')] $*"; }
while :; do
  log "launching segment (restart #$restarts): $*"
  $PY -u phase4/40_run.py --run-name $RUN "$@" >> logs/phase4/$RUN.log 2>&1 &
  pid=$!; echo $pid > logs/phase4/$RUN.pid; last_size=-1; last_change=$(date +%s); stale=0
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    size=$(stat -c %s "$PROGRESS" 2>/dev/null || echo 0)
    if [ "$size" != "$last_size" ]; then last_size=$size; last_change=$(date +%s); fi
    # evals + sampling + checkpoint writes pause the step log for a few minutes; STALE_SEC must exceed that
    if [ $(( $(date +%s) - last_change )) -gt $STALE_SEC ]; then
      log "STALE: no progress for ${STALE_SEC}s -> kill -9 $pid"; kill -9 $pid; stale=1; sleep 10
      journalctl -k --since "-${STALE_SEC}s" --no-pager 2>/dev/null | grep -E "Timedout|Engine reset" | tail -2; break
    fi
  done
  wait $pid; rc=$?
  if [ $rc -eq 0 ] && [ $stale -eq 0 ]; then log "segment ended cleanly: $(tail -1 logs/phase4/$RUN.log | cut -c1-160)"; break; fi
  restarts=$((restarts+1)); log "exited rc=$rc"
  if [ $rc -eq 3 ]; then log "NaN loss: not restarting"; exit 3; fi
  if [ $restarts -ge $MAX_RESTARTS ]; then log "giving up after $restarts restarts"; exit 1; fi
  sleep 20
done
rm -f logs/phase4/$RUN.pid
