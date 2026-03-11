#!/bin/bash

# Usage: ./evals/segmentation/search_lr.sh <script_name> <ckpt_path> <output_dir> <max_res> [feature_type] [extra_args...]
# Example: ./evals/segmentation/search_lr.sh ade20k checkpoints/model.pt outputs/ade20k 256 amoe --root_dir /path/to/ade20k

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <script_name> <ckpt_path> <output_dir> <max_res> [feature_type] [extra_args...]"
    echo "Example: $0 ade20k checkpoints/model.pt outputs/ade20k 256 amoe --root_dir /data/ade20k"
    exit 1
fi


SCRIPT_NAME=$1
CKPT_PATH=$2
OUTPUT_DIR=$3
MAX_RES=$4
FEATURE_TYPE=${5:-"amoe"}
shift 5 
EXTRA_ARGS=$@

# Determine the script path
if [ -f "evals/segmentation/${SCRIPT_NAME}.py" ]; then
    PY_SCRIPT="evals/segmentation/${SCRIPT_NAME}.py"
elif [ -f "${SCRIPT_NAME}.py" ]; then
    PY_SCRIPT="${SCRIPT_NAME}.py"
else
    echo "Error: Script ${SCRIPT_NAME}.py not found in evals/segmentation/ or current directory."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

LR_LIST=(0.0004 0.0008 0.001 0.002 0.004 0.008 0.01 0.02)
NUM_GPUS=$(nvidia-smi -L | wc -l 2>/dev/null || echo 1)
CKPT_NAME=$(basename "$CKPT_PATH")

echo "Searching LR for $SCRIPT_NAME using $NUM_GPUS GPUs..."
echo "Checkpoint: $CKPT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Feature type: $FEATURE_TYPE"
echo "Extra args: $EXTRA_ARGS"

i=0
for LR in "${LR_LIST[@]}"; do
    GPU_ID=$((i % NUM_GPUS))
    
    # Construct the output directory for this LR
    LR_OUT_DIR="${OUTPUT_DIR}/${SCRIPT_NAME}_${FEATURE_TYPE}_lr${LR}_res${MAX_RES}_${CKPT_NAME}"
    
    echo "[LR=$LR] Launching on GPU $GPU_ID"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${PY_SCRIPT} \
        --ckpt_path "${CKPT_PATH}" \
        --out_dir "${LR_OUT_DIR}" \
        --image_size "${MAX_RES}" \
        --lr "${LR}" \
        --feature_type "${FEATURE_TYPE}" \
        ${EXTRA_ARGS} &
    
    sleep 2
    i=$((i+1))
done

wait
echo "All LR search jobs completed."
