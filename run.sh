#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python_bin="${PYTHON:-python3.10}"
if [[ ! -x .venv/bin/python ]]; then
    "$python_bin" -m venv .venv
fi
if [[ ! -f .venv/.requirements-installed ]] || ! cmp -s requirements.txt .venv/.requirements-installed; then
    .venv/bin/python -m pip install -r requirements.txt
    cp requirements.txt .venv/.requirements-installed
fi
exec .venv/bin/python reproduce.py "$@"
