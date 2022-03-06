#!/bin/bash

DATESTR=$(date +"%F-%H-%M-%S")

MODE=$1
TASK=$2
GPU_INDEX=$3
if test $# -eq 3; then
  MODEL_CLS=${TASK}
  MODEL_SAVE_DIR=${TASK}/models/${DATESTR}
elif test $# -eq 4; then
  MODEL_CLS=${TASK}$4
  MODEL_SAVE_DIR=${TASK}/models/$4/${DATESTR}
fi
DATA_CLS=My${TASK}
DATA_DIR=${TASK}/datasets

# fill these for your usage
MODEL_LOAD_PATH=fill/this/for/your/usage

if test "${MODE}" = train; then
  if test "${TASK}" = ModelNet40; then
    python main.py --mode train --task "${TASK}" --batch-size 32 --learning-rate 1e-2 --learning-rate-milestones 200 --soft-cross-entropy 0.2 --max-epoch 250 --model-cls "${MODEL_CLS}" --model-save-dir "${MODEL_SAVE_DIR}" --data-cls "${DATA_CLS}" --data-dir "${DATA_DIR}" --gpu "${GPU_INDEX}" --random-seed 42
  elif test "${TASK}" = ShapeNetPart; then
    python main.py --mode train --task "${TASK}" --batch-size 32 --learning-rate 1e-2 --learning-rate-milestones 200,400  --max-epoch 500 --model-cls "${MODEL_CLS}" --model-save-dir "${MODEL_SAVE_DIR}" --data-cls "${DATA_CLS}" --data-dir "${DATA_DIR}" --gpu "${GPU_INDEX}" --random-seed 318
  fi
elif test "${MODE}" = eval; then
  if test "${TASK}" = ModelNet40; then
    python main.py --mode eval --task "${TASK}" --batch-size 32 --model-cls "${MODEL_CLS}" --model-load-path "${MODEL_LOAD_PATH}" --data-cls "${DATA_CLS}" --data-dir "${DATA_DIR}" --gpu "${GPU_INDEX}"
  elif test "${TASK}" = ShapeNetPart; then
    python main.py --mode eval --task "${TASK}" --batch-size 32 --model-cls "${MODEL_CLS}" --model-load-path "${MODEL_LOAD_PATH}" --data-cls "${DATA_CLS}" --data-dir "${DATA_DIR}" --gpu "${GPU_INDEX}"
  fi
fi
