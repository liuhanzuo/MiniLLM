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
MASTER_ADDR=${MASTER_ADDR:-29.119.84.77}
MASTER_PORT=${MASTER_PORT:-29530}

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

# Optional: Pre-download HF model to a shared path to avoid 429 rate-limit across ranks
# You can override via env: MODEL_ID, HF_REVISION, MODEL_LOCAL_DIR, PRELOAD
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}
HF_REVISION=${HF_REVISION:-main}
MODEL_LOCAL_DIR=${MODEL_LOCAL_DIR:-./hf/${MODEL_ID//\//_}}
PRELOAD=${PRELOAD:-0}
USE_REMOTE_TOKENIZER=${USE_REMOTE_TOKENIZER:-0}

# Export so the inline Python (snapshot_download) can read them via os.environ
export MODEL_ID
export HF_REVISION
export MODEL_LOCAL_DIR
export PRELOAD

# Common HF cache dirs (shared)
export HF_HOME=${HF_HOME:-/apdcephfs/pig/.cache/hf}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export HF_HUB_DISABLE_TELEMETRY=${HF_HUB_DISABLE_TELEMETRY:-1}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

# Ensure we use the current Python env early (needed for preload)
export DS_PYTHON_EXEC=${DS_PYTHON_EXEC:-$(which python3)}
export DS_PYTHON=${DS_PYTHON:-${DS_PYTHON_EXEC}}

if [[ "${PRELOAD}" == "1" ]]; then
  echo "[preload] Downloading ${MODEL_ID}@${HF_REVISION} to ${MODEL_LOCAL_DIR} (once on master)" >&2
  # Ensure HF caches exist
  mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"
  # Temporarily disable offline to allow downloading even if user exported it
  PREV_HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-}
  unset HF_HUB_OFFLINE
  "${DS_PYTHON_EXEC}" - <<'PY'
import os, sys
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

model_id = os.environ.get('MODEL_ID')
revision = os.environ.get('HF_REVISION', 'main')
local_dir = os.environ.get('MODEL_LOCAL_DIR')
token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN') or None
endpoint = os.environ.get('HF_ENDPOINT') or os.environ.get('HF_MIRROR') or None
os.makedirs(local_dir, exist_ok=True)

def looks_complete(d: str) -> bool:
  try:
    entries = set(os.listdir(d))
  except Exception:
    return False
  required = {"config.json", "tokenizer.json"}
  has_model = any(
    name.endswith(('.safetensors', '.bin')) or name in (
      'pytorch_model.bin.index.json', 'model.safetensors'
    ) for name in entries
  )
  return required.issubset(entries) and has_model

if looks_complete(local_dir):
  print(f"[preload] Detected existing local model files in {local_dir}, skip downloading.")
  sys.exit(0)

if endpoint:
  # Point to mirror, e.g. https://hf-mirror.com
  os.environ['HF_ENDPOINT'] = endpoint
  print(f"[preload] Using HF endpoint mirror: {endpoint}")

attempts = 6
delays = [2, 5, 10, 20, 40, 60]
last_err = None
for i in range(attempts):
  try:
    print(f"[preload] snapshot_download({model_id!r}, rev={revision!r}, dst={local_dir}, attempt={i+1}/{attempts})")
    snapshot_download(
      repo_id=model_id,
      revision=revision,
      local_dir=local_dir,
      local_dir_use_symlinks=False,
      token=token,
      resume_download=True,
    )
    print("[preload] Download completed.")
    last_err = None
    break
  except HfHubHTTPError as e:
    last_err = e
    code = getattr(e.response, 'status_code', None)
    if code == 429 and i < attempts - 1:
      wait = delays[i] if i < len(delays) else delays[-1]
      print(f"[preload] Got 429 Too Many Requests, retrying in {wait}s...")
      time.sleep(wait)
      continue
    else:
      break
  except Exception as e:
    last_err = e
    break

if last_err is not None:
  print("[preload] Failed to pre-download from HuggingFace Hub.")
  if isinstance(last_err, HfHubHTTPError):
    code = getattr(last_err.response, 'status_code', None)
    print(f"[preload] HfHubHTTPError: HTTP {code}: {last_err}")
    print("[preload] Suggestions: set HUGGINGFACE_HUB_TOKEN for authenticated higher rate limits, or set HF_ENDPOINT to a mirror, or manually place the model under the target directory and rerun.")
  else:
    print(f"[preload] {type(last_err).__name__}: {last_err}")
  sys.exit(1)
else:
  # Validate critical files exist before proceeding to training
  import json
  missing = []
  for fname in ("config.json", "tokenizer.json"):
    if not os.path.isfile(os.path.join(local_dir, fname)):
      missing.append(fname)
  if missing:
    print(f"[preload] Missing critical files in {local_dir}: {missing}")
    print("[preload] Training aborted. Please retry download with a token or mirror: \n"
        "  export HUGGINGFACE_HUB_TOKEN=...  # if you have one\n"
        "  export HF_ENDPOINT=https://hf-mirror.com  # if available\n"
        "Then rerun the launcher.")
    sys.exit(1)
  # Verify model_type in config.json
  cfg_path = os.path.join(local_dir, 'config.json')
  try:
    with open(cfg_path, 'r') as f:
      cfg = json.load(f)
    mt = cfg.get('model_type')
    if not mt:
      print(f"[preload] config.json in {local_dir} has no 'model_type'. This may break AutoTokenizer/AutoConfig.")
      print("[preload] Please ensure the snapshot is complete and from a supported transformers version, or upgrade transformers.")
      sys.exit(1)
    else:
      print(f"[preload] Detected model_type in config.json: {mt}")
  except Exception as e:
    print(f"[preload] Failed to parse config.json: {e}")
    sys.exit(1)
PY
  # Append overrides so argparse picks the last occurrence
  if [[ "${USE_REMOTE_TOKENIZER}" == "1" ]]; then
    echo "[preload] USE_REMOTE_TOKENIZER=1: will fetch tokenizer from Hub (model local)." >&2
    # Do NOT force offline in this mode, so tokenizer can be fetched
    # User should set HUGGINGFACE_HUB_TOKEN / HF_ENDPOINT to avoid 429.
    TRAIN_ARGS+=" --model_name_or_path ${MODEL_LOCAL_DIR} --tokenizer_dir ${MODEL_ID} --hf_revision ${HF_REVISION}"
  else
    # Prefer offline/local path during training to avoid concurrent network access
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
    TRAIN_ARGS+=" --model_name_or_path ${MODEL_LOCAL_DIR} --tokenizer_dir ${MODEL_LOCAL_DIR} --hf_revision ${HF_REVISION}"
  fi
fi

if [[ -z "${MASTER_ADDR}" || -z "${MASTER_PORT}" ]]; then
  echo "MASTER_ADDR and MASTER_PORT must be set in environment." >&2
  exit 1
fi

if [[ ! -f "${HOSTFILE}" ]]; then
  echo "Hostfile not found: ${HOSTFILE}" >&2
  exit 1
fi

# Recommend setting network interface for NCCL/GLOO
IFACE=${IFACE:-bond1}
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
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
unset NCCL_ASYNC_ERROR_HANDLING
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-PHB}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-23}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7}
export NCCL_CUDA_GRAPH_DISABLE=${NCCL_CUDA_GRAPH_DISABLE:-1}

# Ensure we use the current Python env
export DS_PYTHON_EXEC=${DS_PYTHON_EXEC:-$(which python3)}

echo "Launching DeepSpeed multi-node training:" >&2
echo "  HOSTFILE=${HOSTFILE}" >&2
echo "  MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}" >&2
echo "  NUM_NODES=${NUM_NODES} NUM_GPUS=${NUM_GPUS}" >&2
echo "  IFACE=${IFACE} (GLOO=${GLOO_SOCKET_IFNAME:-} NCCL=${NCCL_SOCKET_IFNAME:-})" >&2
echo "  TRAIN_SCRIPT=${TRAIN_SCRIPT}" >&2
echo "  TRAIN_ARGS=${TRAIN_ARGS}" >&2
echo "  DS_PYTHON_EXEC=${DS_PYTHON_EXEC}" >&2
if [[ "${PRELOAD}" == "1" ]]; then
  echo "  PRELOADED MODEL: ${MODEL_ID}@${HF_REVISION} -> ${MODEL_LOCAL_DIR}" >&2
fi

"${DS_PYTHON_EXEC}" -u -m deepspeed \
  --hostfile "${HOSTFILE}" \
  --num_nodes "${NUM_NODES}" \
  --num_gpus "${NUM_GPUS}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" ${TRAIN_ARGS}
