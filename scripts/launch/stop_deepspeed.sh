#!/usr/bin/env bash
set -euo pipefail

# Gracefully stop a DeepSpeed multi-node job launched via deepspeed CLI or our wrapper.
# It sends SIGINT, waits, then SIGTERM, and finally SIGKILL to remaining matching processes on all hosts.
#
# Usage:
#   export HOSTFILE=scripts/launch/hostfile.example
#   # PATTERN defaults to training script path; customize if needed
#   export PATTERN="class/lec4/train_distillation.py"
#   bash scripts/launch/stop_deepspeed.sh
#
# Notes:
# - Requires passwordless SSH to all hosts in HOSTFILE.
# - PATTERN should uniquely match your training python command (avoid killing unrelated jobs).

HOSTFILE=${HOSTFILE:-scripts/launch/hostfile.example}
PATTERN=${PATTERN:-class/lec4/train_distillation.py}

if [[ -z "${HOSTFILE}" || ! -f "${HOSTFILE}" ]]; then
  echo "HOSTFILE not set or not found: ${HOSTFILE}" >&2
  exit 1
fi

# Build host list (comma-separated) from hostfile (ignore comments/blank lines)
HOSTS=$(awk 'NF && $1 !~ /^#/ {print $1}' "${HOSTFILE}" | paste -sd, -)
if [[ -z "${HOSTS}" ]]; then
  echo "No hosts found in ${HOSTFILE}" >&2
  exit 1
fi

echo "Stopping DeepSpeed job on hosts: ${HOSTS}" >&2
echo "Match pattern: ${PATTERN}" >&2

remote_kill() {
  local signal=$1
  pdsh -R ssh -w "${HOSTS}" \
    "pgrep -f '${PATTERN}' >/dev/null && pkill -${signal} -f '${PATTERN}' || true; \
     pgrep -f 'deepspeed.launcher|deepspeed/launcher|torch.distributed.run|torchrun' >/dev/null && pkill -${signal} -f 'deepspeed.launcher|deepspeed/launcher|torch.distributed.run|torchrun' || true" || true
}

remote_ps() {
  pdsh -R ssh -w "${HOSTS}" "ps -eo pid,ppid,cmd | egrep -E '${PATTERN}|deepspeed.launcher|torchrun|torch.distributed.run' | grep -v egrep || true" || true
}

echo "[phase 1] Sending SIGINT (Ctrl-C) ..." >&2
remote_kill INT
sleep 5
echo "Remaining processes (if any):" >&2
remote_ps || true

echo "[phase 2] Sending SIGTERM ..." >&2
remote_kill TERM
sleep 5
echo "Remaining processes (if any):" >&2
remote_ps || true

echo "[phase 3] Sending SIGKILL ..." >&2
remote_kill KILL
sleep 2
echo "Final check:" >&2
remote_ps || true

echo "Stop sequence completed." >&2
