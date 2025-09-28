#!/usr/bin/env bash
set -euo pipefail
# 2 nodes x 8 GPUs Distillation launcher using deepspeed_2n8g.sh
# Supports MiniLLM teacher or HF teacher via TEACHER_TYPE env
# Usage: export HOSTFILE, MASTER_ADDR, MASTER_PORT; then bash scripts/launch/distill_2n8g.sh

export NUM_NODES=${NUM_NODES:-2}
export NUM_GPUS=${NUM_GPUS:-8}
export TRAIN_SCRIPT=./scripts/train/train_distillation.py

# Student
S_DIM=${S_DIM:-1024}
S_LAYERS=${S_LAYERS:-24}
MAXLEN=${MAXLEN:-4096}
BATCH=${BATCH:-2}
ACC=${ACC:-8}
EPOCHS=${EPOCHS:-2}
LR=${LR:-5e-6}
TOK_DIR=${TOK_DIR:-./model/minillm_tokenizer}
OUT_DIR=${OUT_DIR:-./out}

# Teacher
TEACHER_TYPE=${TEACHER_TYPE:-hf}  # hf | mini
T_DIM=${T_DIM:-2048}
T_LAYERS=${T_LAYERS:-24}
T_CKP=${T_CKP:-./out/full_sft_${T_DIM}.pth}
T_MODEL_ID=${T_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}

BASE_ARGS="\
  --out_dir ${OUT_DIR} \
  --epochs ${EPOCHS} --batch_size ${BATCH} --learning_rate ${LR} \
  --data_path ./dataset/sft_data.jsonl \
  --tokenizer_dir ${TOK_DIR} \
  --student_dim ${S_DIM} --student_layers ${S_LAYERS} --max_seq_len ${MAXLEN} \
  --distillation_mode logit --temperature 2.0 --alpha 0.3 \
  --dtype bfloat16"

if [[ "${TEACHER_TYPE}" == "hf" ]]; then
  export TRAIN_ARGS="${BASE_ARGS} \
    --teacher_model_type hf \
    --teacher_model_name_or_path ${T_MODEL_ID} \
    --trust_remote_code"
else
  export TRAIN_ARGS="${BASE_ARGS} \
    --teacher_dim ${T_DIM} --teacher_layers ${T_LAYERS} \
    --teacher_ckpt ${T_CKP}"
fi

bash scripts/launch/deepspeed_2n8g.sh
