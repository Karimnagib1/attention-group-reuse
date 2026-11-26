#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

GRACEFUL_STOP_EXIT_CODE=75
REQUEUE_REQUESTED=0
child_pid=""
SCONTROL_BIN="${SCONTROL_BIN:-scontrol}"

forward_slurm_warning() {
  REQUEUE_REQUESTED=1
  echo "SLURM wall-time warning received; requesting a checkpoint stop."
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill -USR1 "${child_pid}"
  fi
}

trap forward_slurm_warning USR1

set +e
"$@" &
child_pid=$!
while true; do
  wait "${child_pid}"
  child_status=$?
  if kill -0 "${child_pid}" 2>/dev/null; then
    continue
  fi
  break
done
set -e
trap - USR1

if [[ "${child_status}" -eq "${GRACEFUL_STOP_EXIT_CODE}" ]]; then
  if [[ "${REQUEUE_REQUESTED}" -ne 1 ]]; then
    echo "Runner requested a resumable stop without a SLURM warning; checkpoint is safe for manual resubmission." >&2
    exit "${child_status}"
  fi
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Checkpoint is safe, but no SLURM job ID is available for automatic requeue." >&2
    exit "${child_status}"
  fi

  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]] && [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    requeue_target="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  else
    requeue_target="${SLURM_JOB_ID}"
  fi
  echo "Requeueing SLURM task ${requeue_target} from its latest checkpoint."
  "${SCONTROL_BIN}" requeue "${requeue_target}"
  exit 0
fi

exit "${child_status}"
