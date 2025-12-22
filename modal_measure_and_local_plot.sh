#!/usr/bin/env bash
set -euo pipefail

MODAL_VOLUME="${MODAL_VOLUME:-ping-llm}"
RUN_DIR="outputs/paper_metrics/default"
MODAL_ARGS=()

if [[ $# -gt 0 ]]; then
  if [[ "$1" != "--" ]]; then
    RUN_DIR="$1"
    shift
  fi
  if [[ "${1:-}" == "--" ]]; then
    shift
    MODAL_ARGS=("$@")
  fi
fi

if [[ "$RUN_DIR" = /* ]]; then
  echo "RUN_DIR must be workspace-relative (e.g., outputs/paper_metrics/default)" >&2
  exit 1
fi

MODAL_RUN_DIR="/mnt/${RUN_DIR}"
REMOTE_RUN_DIR="${RUN_DIR}"

echo "Running Modal evaluation -> ${MODAL_RUN_DIR}"
modal run scripts/eval_paper_metrics.py::eval_on_modal \
  --output-dir "${MODAL_RUN_DIR}" \
  "${MODAL_ARGS[@]}"

echo "Pulling run data from Modal volume '${MODAL_VOLUME}'"
if [[ -e "${RUN_DIR}" ]]; then
  rm -rf "${RUN_DIR}"
fi
mkdir -p "$(dirname "${RUN_DIR}")"
modal volume get "${MODAL_VOLUME}" "${REMOTE_RUN_DIR}" "$(dirname "${RUN_DIR}")"

echo "Rendering plots locally"
python scripts/eval_paper_metrics_plot.py --run-dir "${RUN_DIR}"
