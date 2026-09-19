#!/usr/bin/env bash
# Phase 5 driver: runs the sessions STRICTLY one at a time (never two model stacks on the card), each with a stall guard.
# Sessions resume by key, so a wedge or crash costs only the in-flight generation. Usage: bench/run_all.sh [sessions...]
cd "$(dirname "$0")/.."; PY=.venv/bin/python; STALE_SEC=${STALE_SEC:-1500}; MAX_RESTARTS=${MAX_RESTARTS:-5}
SESSIONS=("$@"); [ ${#SESSIONS[@]} -eq 0 ] && SESSIONS=(main q swap gsm gsmq)
mkdir -p logs/phase5; log(){ echo "[bench $(date '+%F %T')] $*"; }
for s in "${SESSIONS[@]}"; do
  restarts=0; raw=logs/phase5/raw_$s.jsonl
  while :; do
    log "session $s launch (restart #$restarts)"
    $PY -u -m bench.run_bench --session $s >> logs/phase5/$s.log 2>&1 &
    pid=$!; last_size=-1; last_change=$(date +%s); stale=0
    while kill -0 $pid 2>/dev/null; do
      sleep 30; size=$(stat -c %s "$raw" 2>/dev/null || echo 0)
      if [ "$size" != "$last_size" ]; then last_size=$size; last_change=$(date +%s); fi
      if [ $(( $(date +%s) - last_change )) -gt $STALE_SEC ]; then log "STALE ${STALE_SEC}s -> kill -9 $pid"; kill -9 $pid; stale=1; sleep 15; break; fi
    done
    wait $pid; rc=$?
    if [ $rc -eq 0 ] && [ $stale -eq 0 ] && tail -5 logs/phase5/$s.log | grep -q "SESSION COMPLETE"; then log "session $s complete"; break; fi
    restarts=$((restarts+1)); log "session $s exited rc=$rc stale=$stale"
    if [ $restarts -ge $MAX_RESTARTS ]; then log "session $s: giving up after $restarts restarts"; break; fi
    sleep 30
  done
done
log "ALL DONE"
