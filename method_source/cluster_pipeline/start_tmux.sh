#!/usr/bin/env bash
# Resume scenario modeling and its read-only dashboard.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(dirname -- "$SCRIPT_DIR")
PYTHON=${PYTHON:-/home/lb/venvs/foredroid/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/lb/data/Dataset/ForeDroid_output/2014-2017}
MODELS=${MODELS:-/home/lb/models/foredroid}
JOBS=${JOBS:-32}
NLP_WORKERS=${NLP_WORKERS:-4}
SESSION=${SESSION:-foredroid_batch}
has_window() {
    local name
    while IFS= read -r name; do
        [[ $name == "$1" ]] && return 0
    done < <(tmux list-windows -t "$SESSION" -F '#{window_name}')
    return 1
}
mkdir -p "$OUTPUT_ROOT/cluster_pipeline"
printf -v run_cmd '%q ' "$PYTHON" -u "$SCRIPT_DIR/run_scenarios.py" --root "$OUTPUT_ROOT" --models "$MODELS" --jobs "$JOBS" --nlp-workers "$NLP_WORKERS"
printf -v log_path '%q' "$OUTPUT_ROOT/cluster_pipeline/run.log"
run_cmd="set -o pipefail; ${run_cmd}2>&1 | tee -a $log_path"
printf -v run_shell '%q ' /bin/bash -lc "$run_cmd"
if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n scenarios -c "$PROJECT_DIR" -x 120 -y 40 "$run_shell"
elif has_window scenarios; then
    pane_dead=$(tmux display-message -p -t "$SESSION:scenarios" '#{pane_dead}')
    if [[ $pane_dead == 1 ]]; then
        tmux respawn-pane -k -t "$SESSION:scenarios" "$run_shell"
    else
        echo 'Scenario window is already running; keeping its current task.'
    fi
else
    tmux new-window -d -t "$SESSION:" -n scenarios -c "$PROJECT_DIR" "$run_shell"
fi
tmux set-window-option -t "$SESSION:scenarios" remain-on-exit on
printf -v monitor_cmd '%q ' "$PYTHON" "$PROJECT_DIR/monitor_clusters.py" --output "$OUTPUT_ROOT"
if has_window scenarios_monitor; then
    tmux respawn-pane -k -t "$SESSION:scenarios_monitor" "$monitor_cmd"
else
    tmux new-window -d -t "$SESSION:" -n scenarios_monitor -c "$PROJECT_DIR" "$monitor_cmd"
fi
tmux set-window-option -t "$SESSION:scenarios_monitor" remain-on-exit on
tmux select-window -t "$SESSION:scenarios_monitor"
echo "Started/resumed scenarios; monitor: tmux attach -t $SESSION (scenarios_monitor window)"
