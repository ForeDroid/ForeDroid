#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(dirname -- "$SCRIPT_DIR")
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/lb/data/Dataset/ForeDroid_output/2018-2022}
YEARS=${YEARS:-2018 2019 2020 2021 2022}
SESSION=${SESSION:-foredroid_testing}
PYTHON=${PYTHON:-/home/lb/venvs/foredroid/bin/python}
FUSION_WORKERS=${FUSION_WORKERS:-6}
DESCRIPTION_WORKERS=${DESCRIPTION_WORKERS:-32}
REUSE_CACHE=${REUSE_CACHE:-/home/lb/data/Dataset/ForeDroid_output/2014-2017/descriptions.sqlite}
read -r -a year_args <<< "$YEARS"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "Existing session retained: tmux attach -t $SESSION"
    exit 0
fi
[[ -d "$OUTPUT_ROOT" ]] || { echo "Missing extraction output: $OUTPUT_ROOT" >&2; exit 1; }
flock -n "$OUTPUT_ROOT/run.lock" true || { echo 'Another controller holds the output lock' >&2; exit 1; }
printf -v fusion '%q ' env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" -u "$SCRIPT_DIR/pipeline.py" --output-root "$OUTPUT_ROOT" --years "${year_args[@]}" --workers "$FUSION_WORKERS"
printf -v descriptions '%q ' env PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -u "$SCRIPT_DIR/generate_description.py" --output-root "$OUTPUT_ROOT" --splits test --model MiniMax-M3 --workers "$DESCRIPTION_WORKERS" --reuse-cache "$REUSE_CACHE"
printf -v logfile '%q' "$OUTPUT_ROOT/pipeline.log"
printf -v cmd '%q ' /bin/bash -lc "set -o pipefail; { $fusion && $descriptions; } 2>&1 | tee -a $logfile"
tmux new-session -d -s "$SESSION" -n pipeline -c "$PROJECT_DIR" "$cmd"
tmux set-window-option -t "$SESSION:pipeline" remain-on-exit on
printf -v monitor '%q ' "$PYTHON" "$PROJECT_DIR/monitor_pipeline.py" --output "$OUTPUT_ROOT" --years "${year_args[@]}"
tmux new-window -d -t "$SESSION:" -n fusion_monitor -c "$PROJECT_DIR" "$monitor"
printf -v monitor '%q ' "$PYTHON" "$PROJECT_DIR/monitor_descriptions.py" --output "$OUTPUT_ROOT"
tmux new-window -d -t "$SESSION:" -n description_monitor -c "$PROJECT_DIR" "$monitor"
tmux select-window -t "$SESSION:fusion_monitor"
echo "Started fusion then M3 descriptions: tmux attach -t $SESSION"
