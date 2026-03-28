#!/bin/bash

set -euo pipefail

CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/opt/miniconda3"
fi

if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate my_env
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/planner_toolbox_mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

if [ -z "${MPLBACKEND:-}" ]; then
    if python3 - <<'PY' >/dev/null 2>&1
import tkinter
PY
    then
        export MPLBACKEND="TkAgg"
    fi
fi

python3 planner_toolbox.py log "$@"

exec bash
