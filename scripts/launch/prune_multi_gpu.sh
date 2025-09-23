#!/usr/bin/env bash
# Multi-GPU pruning launcher using torchrun
# Usage example:
#   bash prune_multi_gpu.sh --nproc_per_node 8 --model_name_or_path qwen/Qwen2.5-7b-Instruct --output_dir /path/to/out --sparsity 0.9

set -euo pipefail

# defaults
NPROC_PER_NODE=1
MASTER_PORT=12355
MODEL_NAME_OR_PATH=""
OUTPUT_DIR=""
SPARSITY=0.9
DEVICE="cuda"
ADDITIONAL_ARGS=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --nproc_per_node)
      NPROC_PER_NODE="$2"; shift; shift;;
    --master_port)
      MASTER_PORT="$2"; shift; shift;;
    --model_name_or_path)
      MODEL_NAME_OR_PATH="$2"; shift; shift;;
    --output_dir)
      OUTPUT_DIR="$2"; shift; shift;;
    --sparsity)
      SPARSITY="$2"; shift; shift;;
    --device)
      DEVICE="$2"; shift; shift;;
    --)
      shift; ADDITIONAL_ARGS="$@"; break;;
    *)
      ADDITIONAL_ARGS+="$1 "; shift;;
  esac
done

if [[ -z "$MODEL_NAME_OR_PATH" || -z "$OUTPUT_DIR" ]]; then
  echo "--model_name_or_path and --output_dir are required"
  exit 1
fi

export TORCH_DISTRIBUTED_DEBUG=INFO

# Launch with torchrun. The pruning script is not distributed-aware, but loading on each rank
# and pruning redundantly is safe for relatively small models; alternatively you can run on
# a single rank (nproc_per_node=1) with larger memory. If you want rank 0 to do the save, we
# implement a simple barrier using torch.distributed if needed inside the python script.

echo "Starting torchrun with ${NPROC_PER_NODE} procs per node, master_port=${MASTER_PORT}"

torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} \
  ../prune_qwen.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --sparsity "${SPARSITY}" \
  --device "${DEVICE}" ${ADDITIONAL_ARGS}

