#!/usr/bin/env bash
set -euo pipefail
# 2 nodes x 8 GPUs pretrain launcher using deepspeed_2n8g.sh
# Usage:
#   export HOSTFILE=scripts/launch/hostfile.example
#   export MASTER_ADDR=10.0.0.1
#   export MASTER_PORT=29530
#   bash scripts/launch/pretrain_2n8g.sh
# Memory tips:
#   - If OOM on 80GB: try BATCH=1 ACC=32 MAXLEN=1536 DTYPE=bfloat16 GRAD_CP=1
#   - If OOM on 40GB: try BATCH=1 ACC=48 MAXLEN=1024 DTYPE=bfloat16 GRAD_CP=1

export NUM_NODES=${NUM_NODES:-2}
export NUM_GPUS=${NUM_GPUS:-8}
# Use absolute path so all nodes import the same codebase
TRAIN_SCRIPT_REL=./scripts/train/train_pretrain.py
export TRAIN_SCRIPT=$(readlink -f "${TRAIN_SCRIPT_REL}")

# Tunables (per-rank batch). Adjust by memory.
DIM=${DIM:-2048}
LAYERS=${LAYERS:-24}
BLOCK=${BLOCK:-None}
MAXLEN=${MAXLEN:-2048}
BATCH=${BATCH:-8}
EPOCHS=${EPOCHS:-3}
LR=${LR:-5e-4}
DATA_PATH=${DATA_PATH:-./dataset/pretrain_hq.jsonl}
TOKENIZER_DIR=${TOKENIZER_DIR:-./model/minillm_tokenizer}
OUT_DIR=${OUT_DIR:-./out}
DTYPE=${DTYPE:-bfloat16}
LOG_INT=${LOG_INT:-100}
SAVE_INT=${SAVE_INT:-500}
ACC=${ACC:-8}
GRAD_CP=${GRAD_CP:-1}

export TRAIN_ARGS="\
  --out_dir ${OUT_DIR} \
  --epochs ${EPOCHS} --batch_size ${BATCH} --learning_rate ${LR} \
  --dim ${DIM} --n_layers ${LAYERS} --n_block ${BLOCK} --max_seq_len ${MAXLEN} \
  --data_path ${DATA_PATH} \
  --tokenizer_dir ${TOKENIZER_DIR} \
  --dtype ${DTYPE} \
  --accumulation_steps ${ACC} \
  --log_interval ${LOG_INT} \
  --save_interval ${SAVE_INT} \
  --ddp"
N_BLOCK=${N_BLOCK:-${BLOCK:-}}
if [[ -n "${N_BLOCK}" ]]; then
  export TRAIN_ARGS="${TRAIN_ARGS} --n_block ${N_BLOCK}"
fi

if [[ "${GRAD_CP}" == "1" ]]; then
  export TRAIN_ARGS="${TRAIN_ARGS} --gradient_checkpointing"
fi

bash scripts/launch/deepspeed_2n8g.sh
