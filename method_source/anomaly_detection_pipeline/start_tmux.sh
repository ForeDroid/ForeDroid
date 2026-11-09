#!/usr/bin/env bash
# Resume anomaly modeling and its read-only dashboard.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(dirname -- "$SCRIPT_DIR")
PYTHON=${PYTHON:-/home/lb/venvs/foredroid/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/lb/data/Dataset/ForeDroid_output/2014-2017}
MODEL=${MODEL:-/home/lb/models/foredroid/all-MiniLM-L6-v2}
JOBS=${JOBS:-32}
SCORE_WORKERS=${SCORE_WORKERS:-8}
SESSION=${SESSION:-foredroid_batch}
has_window() {
    local name
    while IFS= read -r name; do
        [[ $name == "$1" ]] && return 0
    done < <(tmux list-windows -t "$SESSION" -F '#{window_name}')
    return 1
}
mkdir -p "$OUTPUT_ROOT/anomaly_detection_pipeline"
printf -v run_cmd '%q ' "$PYTHON" -u "$SCRIPT_DIR/run_pipeline.py" --root "$OUTPUT_ROOT" --model "$MODEL" --jobs "$JOBS" --workers "$SCORE_WORKERS"
printf -v log_path '%q' "$OUTPUT_ROOT/anomaly_detection_pipeline/run.log"
run_cmd="set -o pipefail; ${run_cmd}2>&1 | tee -a $log_path"
printf -v run_shell '%q ' /bin/bash -lc "$run_cmd"
if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n anomaly -c "$PROJECT_DIR" -x 120 -y 40 "$run_shell"
elif has_window anomaly; then
    pane_dead=$(tmux display-message -p -t "$SESSION:anomaly" '#{pane_dead}')
    if [[ $pane_dead == 1 ]]; then
        tmux respawn-pane -k -t "$SESSION:anomaly" "$run_shell"
    else
        echo 'Anomaly window is already running; keeping its current task.'
    fi
else
    tmux new-window -d -t "$SESSION:" -n anomaly -c "$PROJECT_DIR" "$run_shell"
fi
tmux set-window-option -t "$SESSION:anomaly" remain-on-exit on
printf -v monitor_cmd '%q ' "$PYTHON" "$PROJECT_DIR/monitor_anomaly.py" --output "$OUTPUT_ROOT"
if has_window anomaly_monitor; then
    tmux respawn-pane -k -t "$SESSION:anomaly_monitor" "$monitor_cmd"
else
    tmux new-window -d -t "$SESSION:" -n anomaly_monitor -c "$PROJECT_DIR" "$monitor_cmd"
fi
tmux set-window-option -t "$SESSION:anomaly_monitor" remain-on-exit on
tmux select-window -t "$SESSION:anomaly_monitor"
echo "Started/resumed anomaly; monitor: tmux attach -t $SESSION (anomaly_monitor window)"
