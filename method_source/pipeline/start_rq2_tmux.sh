#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(dirname -- "$SCRIPT_DIR")
PYTHON=${PYTHON:-/home/lb/venvs/foredroid/bin/python}
SESSION=${RQ2_SESSION:-foredroid_rq2}
ROOT=${RQ2_OUTPUT_ROOT:-/home/lb/data/Dataset/ForeDroid_output/RQ2/GPMalware}
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Session retained: tmux attach -t $SESSION"
  exit 0
fi
mkdir -p "$ROOT"
printf -v job '%q ' "$PYTHON" -u "$SCRIPT_DIR/rq2_preprocess.py" --root "$ROOT"
printf -v log '%q' "$ROOT/preprocess.log"
printf -v shell '%q ' /bin/bash -lc "set -o pipefail; ${job}2>&1 | tee -a $log"
tmux new-session -d -s "$SESSION" -n preprocess -c "$PROJECT_DIR" "$shell"
tmux set-window-option -t "$SESSION:preprocess" remain-on-exit on
printf -v monitor '%q ' "$PYTHON" "$PROJECT_DIR/monitor_rq2.py" --output "$ROOT"
tmux new-window -d -t "$SESSION:" -n monitor -c "$PROJECT_DIR" "$monitor"
tmux select-window -t "$SESSION:monitor"
echo "Started RQ2 preprocessing: tmux attach -t $SESSION"
