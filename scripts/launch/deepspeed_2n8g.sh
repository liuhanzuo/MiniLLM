#!/usr/bin/env bash
set -euo pipefail

# DeepSpeed two-node, eight-GPU-per-node launcher (uses deepspeed CLI)
# Requirements:
#  - deepspeed installed in this Python env (pip install deepspeed)
#  - passwordless SSH among nodes
#  - a hostfile listing nodes and slots (GPUs)
#
# Usage (run on the MASTER node only):
#   export HOSTFILE=scripts/launch/hostfile.example   # or your own
#   export MASTER_ADDR=29.119.84.77                   # IP of this master node
#   export MASTER_PORT=29530                          # free TCP port
#   bash scripts/launch/deepspeed_2n8g.sh

# Config (override via env)
NUM_NODES=${NUM_NODES:-2}
NUM_GPUS=${NUM_GPUS:-8}
HOSTFILE=${HOSTFILE:-scripts/launch/hostfile.example}
MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-}

# Training script and default args for HF SFT (override via env TRAIN_SCRIPT / TRAIN_ARGS)
TRAIN_SCRIPT=${TRAIN_SCRIPT:-./scripts/train/train_full_sft.py}
TRAIN_ARGS=${TRAIN_ARGS:-"\
  --model_type hf \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --tokenizer_dir Qwen/Qwen2.5-7B-Instruct \
  --trust_remote_code \
  --hf_revision main \
  --data_path ./dataset/sft_2048.jsonl \
  --epochs 1 \
  --batch_size 1 \
  --accumulation_steps 16 \
  --learning_rate 2e-5 \
  --max_seq_len 2048 \
  --dtype bfloat16 \
  --device cuda \
  --grad_clip 1.0 \
  --out_dir ./out/qwen2.5_8b_sft \
  --log_interval 10 \
  --save_interval 200 \
  --save_hf \
  --gradient_checkpointing \
  --deepspeed scripts/launch/ds_zero2_30b.json"}

if [[ -z "${MASTER_ADDR}" || -z "${MASTER_PORT}" ]]; then
  echo "MASTER_ADDR and MASTER_PORT must be set in environment." >&2
  exit 1
fi

if [[ ! -f "${HOSTFILE}" ]]; then
  echo "Hostfile not found: ${HOSTFILE}" >&2
  exit 1
fi

# Recommend setting network interface for NCCL/GLOO
IFACE=${IFACE:-}
if [[ -z "${IFACE}" ]]; then
  # Try to infer by routing to the first host in hostfile (if not localhost)
  FIRST_HOST=$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "${HOSTFILE}")
  if [[ -n "${FIRST_HOST}" && "${FIRST_HOST}" != "localhost" && "${FIRST_HOST}" != "127.0.0.1" ]]; then
    IFACE=$(ip route get "${FIRST_HOST}" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}') || true
  fi
fi

if [[ -n "${IFACE}" && "${IFACE}" != "lo" ]]; then
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${IFACE}}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${IFACE}}
fi

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

# Ensure we use the current Python env
export DS_PYTHON_EXEC=${DS_PYTHON_EXEC:-$(which python)}

echo "Launching DeepSpeed multi-node training:" >&2
echo "  HOSTFILE=${HOSTFILE}" >&2
echo "  MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}" >&2
echo "  NUM_NODES=${NUM_NODES} NUM_GPUS=${NUM_GPUS}" >&2
echo "  IFACE=${IFACE} (GLOO=${GLOO_SOCKET_IFNAME:-} NCCL=${NCCL_SOCKET_IFNAME:-})" >&2
echo "  TRAIN_SCRIPT=${TRAIN_SCRIPT}" >&2
echo "  TRAIN_ARGS=${TRAIN_ARGS}" >&2

deepspeed \
  --hostfile "${HOSTFILE}" \
  --num_nodes "${NUM_NODES}" \
  --num_gpus "${NUM_GPUS}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" ${TRAIN_ARGS}
