#!/usr/bin/env bash
# Start (or continue) the Phase 4 run for one segment, detached. Usage: phase4/start.sh [--max-hours 20] [--total-tokens 100e6] ...
cd "$(dirname "$0")/.."; RUN=${RUN:-phase4_run}
if pgrep -f "phase4/40_run.py --run-name $RUN" > /dev/null; then echo "already running"; exit 1; fi
mkdir -p logs/phase4; RUN=$RUN setsid nohup phase4/watchdog.sh "$@" >> logs/phase4/$RUN.watchdog.log 2>&1 < /dev/null &
sleep 2; echo "started segment for $RUN; logs: logs/phase4/$RUN.log  status: .venv/bin/python phase4/status.py  stop: phase4/stop.sh"
