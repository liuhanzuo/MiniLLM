#!/usr/bin/env bash
set -euo pipefail
# 2 nodes x 8 GPUs SFT launcher using deepspeed_2n8g.sh
# HF path variant and MiniLLM variant via MODEL_TYPE env
# Usage: export HOSTFILE, MASTER_ADDR, MASTER_PORT; then bash scripts/launch/sft_2n8g.sh

export NUM_NODES=${NUM_NODES:-2}
export NUM_GPUS=${NUM_GPUS:-8}
export TRAIN_SCRIPT=./scripts/train/train_full_sft.py

MODEL_TYPE=${MODEL_TYPE:-hf}  # hf | mini
OUT_DIR=${OUT_DIR:-./out}

if [[ "${MODEL_TYPE}" == "hf" ]]; then
  MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}
  HF_REVISION=${HF_REVISION:-main}
  TOK_DIR=${TOK_DIR:-${MODEL_ID}}
  BATCH=${BATCH:-1}
  ACC=${ACC:-16}
  EPOCHS=${EPOCHS:-1}
  LR=${LR:-5e-6}
  MAXLEN=${MAXLEN:-4096}
  DS_CFG=${DS_CFG:-scripts/launch/ds_zero3_30b.json}
  export TRAIN_ARGS="\
    --model_type hf \
    --model_name_or_path ${MODEL_ID} \
    --tokenizer_dir ${TOK_DIR} \
    --trust_remote_code \
    --out_dir ${OUT_DIR} \
    --epochs ${EPOCHS} --batch_size ${BATCH} --learning_rate ${LR} \
    --max_seq_len ${MAXLEN} \
    --dtype bfloat16 \
    --gradient_checkpointing \
    --save_hf \
    --deepspeed ${DS_CFG}"
else
  DIM=${DIM:-1024}
  LAYERS=${LAYERS:-24}
  MAXLEN=${MAXLEN:-2048}
  BATCH=${BATCH:-8}
  ACC=${ACC:-4}
  EPOCHS=${EPOCHS:-2}
  LR=${LR:-5e-5}
  CKP=${CKP:-./out/pretrain_${DIM}.pth}
  TOKENIZER_DIR=${TOK_DIR:-./model/minillm_tokenizer}
  BLOCK=${BLOCK:-None}
  export TRAIN_ARGS="\
    --model_type mini \
    --ckp ${CKP} \
    --out_dir ${OUT_DIR} \
    --epochs ${EPOCHS} --batch_size ${BATCH} --learning_rate ${LR} \
    --dim ${DIM} --n_layers ${LAYERS} --max_seq_len ${MAXLEN} \
    --n_block ${BLOCK} \
    --data_path ./dataset/sft_mini_512.jsonl \
    --tokenizer_dir ${TOKENIZER_DIR} \
    --dtype bfloat16 \
    --accumulation_steps ${ACC} \
    --ddp"
fi

bash scripts/launch/deepspeed_2n8g.sh
