#!/usr/bin/env bash
set -euo pipefail
# 2 nodes x 8 GPUs DPO (RLHF) launcher using deepspeed_2n8g.sh (runs torch DDP path)
# Usage: export HOSTFILE, MASTER_ADDR, MASTER_PORT; then bash scripts/launch/dpo_2n8g.sh

export NUM_NODES=${NUM_NODES:-2}
export NUM_GPUS=${NUM_GPUS:-8}
export TRAIN_SCRIPT=./scripts/train/train_dpo.py

DIM=${DIM:-1024}
LAYERS=${LAYERS:-24}
MAXLEN=${MAXLEN:-3000}
BATCH=${BATCH:-2}
ACC=${ACC:-8}
EPOCHS=${EPOCHS:-2}
LR=${LR:-1e-8}
TOK_DIR=${TOK_DIR:-./model/minillm_tokenizer}
OUT_DIR=${OUT_DIR:-./out}

export TRAIN_ARGS="\
  --out_dir ${OUT_DIR} \
  --epochs ${EPOCHS} --batch_size ${BATCH} --learning_rate ${LR} \
  --dim ${DIM} --n_layers ${LAYERS} --max_seq_len ${MAXLEN} \
  --data_path ./dataset/dpo.jsonl \
  --tokenizer_dir ${TOK_DIR} \
  --dtype bfloat16 \
  --accumulation_steps ${ACC} \
  --ddp"

bash scripts/launch/deepspeed_2n8g.sh
