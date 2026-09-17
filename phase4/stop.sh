#!/usr/bin/env bash
# Ask the run to stop cleanly (finishes the current step, saves a full checkpoint, frees the GPU). Resume with phase4/start.sh.
cd "$(dirname "$0")/.."; RUN=${RUN:-phase4_run}
if ! pgrep -f "phase4/40_run.py --run-name $RUN" > /dev/null; then echo "not running"; exit 0; fi
touch checkpoints/$RUN/STOP; echo -n "stop requested; waiting for checkpoint"
for i in $(seq 1 120); do pgrep -f "phase4/40_run.py --run-name $RUN" > /dev/null || { echo; echo "stopped. GPU is free."; tail -1 logs/phase4/$RUN.log; exit 0; }; echo -n .; sleep 2; done
echo; echo "still running after 4 min (mid-eval?). Check logs/phase4/$RUN.log"
