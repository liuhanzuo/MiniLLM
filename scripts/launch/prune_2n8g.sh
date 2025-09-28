#!/usr/bin/env bash
set -euo pipefail
# 2 nodes x 8 GPUs pruning launcher using torchrun on each node via hostfile
# For large HF models. Each rank prunes redundantly; rank0 saves by default.

HOSTFILE=${HOSTFILE:-scripts/launch/hostfile.example}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29531}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}
OUT_DIR=${OUT_DIR:-./out/pruned-model}
SPARSITY=${SPARSITY:-0.9}
EXTRA_ARGS=${EXTRA_ARGS:-"--device_map auto --dtype bfloat16 --include_embeddings --save_state_dict --rank0_only"}

# We will iterate hostfile and run torchrun via pdsh/parallel ssh, but simplest is deepspeed's hostfile style launcher.
# Reuse deepspeed_2n8g.sh infra by setting TRAIN_SCRIPT to pruning script and using TRAIN_ARGS.

export NUM_NODES=${NUM_NODES:-2}
export NUM_GPUS=${NUM_GPUS:-${NPROC_PER_NODE}}
export TRAIN_SCRIPT=./scripts/prune_qwen.py
export TRAIN_ARGS="\
  --model_name_or_path ${MODEL_ID} \
  --output_dir ${OUT_DIR} \
  --sparsity ${SPARSITY} ${EXTRA_ARGS}"

# We do not need deepspeed engine; launcher will still distribute processes per node.
bash scripts/launch/deepspeed_2n8g.sh
