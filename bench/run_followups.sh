#!/usr/bin/env bash
# Wait for a running benchmark process (by PID) to exit, then run the remaining sessions sequentially. Each resumes by key.
cd "$(dirname "$0")/.."; PY=.venv/bin/python; WAIT_PID=${1:-}
log(){ echo "[followups $(date '+%F %T')] $*"; }
if [ -n "$WAIT_PID" ]; then log "waiting for pid $WAIT_PID"; while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done; fi
for s in q swap gsm gsmq; do
  log "session $s start"; $PY -u -m bench.run_bench --session $s >> logs/phase5/$s.log 2>&1; log "session $s exit rc=$?"
done
log "all sessions done"
