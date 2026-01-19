#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/conda-env/yolov26/bin/python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_SRC="${DATASET_SRC:-${REPO_DIR}/Datasets/M3FD}"
DATASET_OUT="${DATASET_OUT:-${REPO_DIR}/Datasets/M3FD_split}"
YAML_OUT="${YAML_OUT:-${REPO_DIR}/m3fd_dual.yaml}"
TRAIN_COUNT="${TRAIN_COUNT:-200}"
VAL_COUNT="${VAL_COUNT:-50}"
SEED="${SEED:-42}"

MODEL_CFG="${MODEL_CFG:-${REPO_DIR}/ultralytics/cfg/models/26/yolo26-dual-transformer.yaml}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-2}"
IMGSZ="${IMGSZ:-320}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-4}"
RUN_DETECT="${RUN_DETECT:-0}"
OPTIMIZER="${OPTIMIZER:-SGD}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found: ${PYTHON}"
  exit 1
fi

if [[ ! -d "${DATASET_SRC}" ]]; then
  echo "Dataset not found: ${DATASET_SRC}"
  exit 1
fi

mkdir -p "${REPO_DIR}/scripts"

"${PYTHON}" "${REPO_DIR}/scripts/prepare_m3fd_subset.py" \
  --src-root "${DATASET_SRC}" \
  --out-root "${DATASET_OUT}" \
  --yaml-out "${YAML_OUT}" \
  --train-count "${TRAIN_COUNT}" \
  --val-count "${VAL_COUNT}" \
  --seed "${SEED}"

"${PYTHON}" - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('device count', torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit('CUDA not available in this Python environment')
PY

cd "${REPO_DIR}"

CUDA_VISIBLE_DEVICES="${DEVICE}" "${PYTHON}" train_dual.py \
  --model "${MODEL_CFG}" \
  --data "${YAML_OUT}" \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --imgsz "${IMGSZ}" \
  --device "${DEVICE}" \
  --optimizer "${OPTIMIZER}" \
  --workers "${WORKERS}" \
  --val

if [[ "${RUN_DETECT}" == "1" ]]; then
  WEIGHTS_PATH="${REPO_DIR}/runs/train/dual_stream_exp/weights/best.pt"
  if [[ -f "${WEIGHTS_PATH}" ]]; then
    CUDA_VISIBLE_DEVICES="${DEVICE}" "${PYTHON}" detect_dual.py \
      --weights "${WEIGHTS_PATH}" \
      --model "${MODEL_CFG}" \
      --rgb-source "${DATASET_OUT}/rgb/val" \
      --ir-source "${DATASET_OUT}/ir/val" \
      --save-dir "${REPO_DIR}/runs/detect/m3fd_test" \
      --device "${DEVICE}"
  else
    echo "Weights not found at ${WEIGHTS_PATH}, skipping detect."
  fi
fi

printf '\nDone. Results in %s\n' "${REPO_DIR}/runs/train/dual_stream_exp"
