#!/bin/bash

# Edit this block for each experiment. Rank 0 is the only rank that uses
# COMPUTE_DEVICE; set it to "cpu" for CPU compute + CUDA/NCCL communication.
WORLD_SIZE_VALUE=${WORLD_SIZE_VALUE:-3}
PREFILL_MODE=${PREFILL_MODE:-distributed}
BATCH_SIZE=${BATCH_SIZE:-1}
SPLIT_LAYERS=${SPLIT_LAYERS:-5,15}
INIT_METHOD=${INIT_METHOD:-tcp://10.50.1.228:29510}
COMPUTE_DEVICE=${COMPUTE_DEVICE:-cuda}
MODEL_DIR=${MODEL_DIR:-/home/dingcong/models/TinyLlama}
INPUT_CSV=${INPUT_CSV:-./dataset/input10.csv}
OUTPUT_CSV=${OUTPUT_CSV:-outputs_kv.csv}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-1000}

RANK_ARG=$1

if [ -z "$RANK_ARG" ]; then
    echo "Usage: ./run.sh 0|1|2"
    exit 1
fi

if [ "$RANK_ARG" = "0" ]; then
    export WORLD_SIZE=$WORLD_SIZE_VALUE
    export RANK=0
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_DEBUG=INFO

    python distributed_tinyllama_inference.py \
      --lazy-load \
      --dynamic-load \
      --prefill-mode $PREFILL_MODE \
      --batch-size $BATCH_SIZE \
      --split-layers $SPLIT_LAYERS \
      --compute-device $COMPUTE_DEVICE \
      --init-method $INIT_METHOD \
      --model-dir $MODEL_DIR \
      --input-csv $INPUT_CSV \
      --output-csv $OUTPUT_CSV \
      --csv-has-header \
      --prompt-column prompt \
      --max-input-tokens $MAX_INPUT_TOKENS
fi

if [ "$RANK_ARG" = "1" ]; then
    export WORLD_SIZE=$WORLD_SIZE_VALUE
    export RANK=1
    export NCCL_SOCKET_IFNAME=enp6s18
    export NCCL_DEBUG=INFO

    python distributed_tinyllama_inference.py \
      --lazy-load \
      --dynamic-load \
      --prefill-mode $PREFILL_MODE \
      --batch-size $BATCH_SIZE \
      --split-layers $SPLIT_LAYERS \
      --init-method $INIT_METHOD \
      --model-dir $MODEL_DIR \
      --csv-has-header \
      --prompt-column prompt \
      --max-input-tokens $MAX_INPUT_TOKENS
fi

if [ "$RANK_ARG" = "2" ]; then
    export WORLD_SIZE=$WORLD_SIZE_VALUE
    export RANK=2
    export NCCL_SOCKET_IFNAME=enp6s18
    export NCCL_DEBUG=INFO

    python distributed_tinyllama_inference.py \
      --lazy-load \
      --dynamic-load \
      --prefill-mode $PREFILL_MODE \
      --batch-size $BATCH_SIZE \
      --split-layers $SPLIT_LAYERS \
      --init-method $INIT_METHOD \
      --model-dir $MODEL_DIR \
      --csv-has-header \
      --prompt-column prompt \
      --max-input-tokens $MAX_INPUT_TOKENS
fi
