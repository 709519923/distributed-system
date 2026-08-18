#!/bin/bash

# Edit this block for each experiment. Rank 0 is the only rank that uses
# COMPUTE_DEVICE; set it to "cpu" for CPU compute + CUDA/NCCL communication.
WORLD_SIZE_VALUE=${WORLD_SIZE_VALUE:-3}
PREFILL_MODE=${PREFILL_MODE:-distributed}
BATCH_SIZE=${BATCH_SIZE:-1}
SPLIT_LAYERS=${SPLIT_LAYERS:-5,15}
SCHEDULER_CSV=${SCHEDULER_CSV:-lipschitz_validation_scheduler.csv}
BANDIT_POLICY=${BANDIT_POLICY:-lipschitz_validation}
INIT_METHOD=${INIT_METHOD:-tcp://10.50.1.130:29510}
COMPUTE_DEVICE=${COMPUTE_DEVICE:-cuda}
MODEL_DIR=${MODEL_DIR:-/home/dingcong/models/TinyLlama}
INPUT_CSV=${INPUT_CSV:-./dataset/lipschitz_validation_prompts.csv}
OUTPUT_CSV=${OUTPUT_CSV:-lipschitz_validation_outputs.csv}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-1000}
FORCE_DECODE_STEPS=${FORCE_DECODE_STEPS:-128}

FORCE_DECODE_STEPS_ARG=""
if [ -n "$FORCE_DECODE_STEPS" ]; then
    FORCE_DECODE_STEPS_ARG="--force-decode-steps $FORCE_DECODE_STEPS"
fi

RANK_ARG=$1

if [ -z "$RANK_ARG" ]; then
    echo "Usage: ./run.sh 0|1|2"
    exit 1
fi

if [ "$RANK_ARG" != "0" ] && [ "$RANK_ARG" != "1" ] && [ "$RANK_ARG" != "2" ]; then
    echo "Usage: ./run.sh 0|1|2"
    exit 1
fi

if [ "$RANK_ARG" = "0" ]; then
    PREFILL_MODE_TEXT="$PREFILL_MODE(rank0-broadcast)"
else
    PREFILL_MODE_TEXT="receive-from-rank0"
fi

echo "[run.sh] RANK=$RANK_ARG WORLD_SIZE=$WORLD_SIZE_VALUE PREFILL_MODE=$PREFILL_MODE_TEXT BATCH_SIZE=$BATCH_SIZE SPLIT_LAYERS=$SPLIT_LAYERS SCHEDULER_CSV=$SCHEDULER_CSV BANDIT_POLICY=$BANDIT_POLICY INIT_METHOD=$INIT_METHOD COMPUTE_DEVICE=$COMPUTE_DEVICE FORCE_DECODE_STEPS=${FORCE_DECODE_STEPS:-off}"

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
      --allocation-csv $SCHEDULER_CSV \
      --bandit-policy $BANDIT_POLICY \
      --compute-device $COMPUTE_DEVICE \
      --init-method $INIT_METHOD \
      --model-dir $MODEL_DIR \
      --input-csv $INPUT_CSV \
      --output-csv $OUTPUT_CSV \
      --csv-has-header \
      --prompt-column prompt \
      --max-input-tokens $MAX_INPUT_TOKENS \
      $FORCE_DECODE_STEPS_ARG
fi

if [ "$RANK_ARG" = "1" ]; then
    export WORLD_SIZE=$WORLD_SIZE_VALUE
    export RANK=1
    export NCCL_SOCKET_IFNAME=enp6s18
    export NCCL_DEBUG=INFO

    python distributed_tinyllama_inference.py \
      --lazy-load \
      --dynamic-load \
      --batch-size $BATCH_SIZE \
      --split-layers $SPLIT_LAYERS \
      --init-method $INIT_METHOD \
      --model-dir $MODEL_DIR \
      --csv-has-header \
      --prompt-column prompt \
      --max-input-tokens $MAX_INPUT_TOKENS \
      $FORCE_DECODE_STEPS_ARG
fi

if [ "$RANK_ARG" = "2" ]; then
    export WORLD_SIZE=$WORLD_SIZE_VALUE
    export RANK=2
    export NCCL_SOCKET_IFNAME=enp6s18
    export NCCL_DEBUG=INFO

    python distributed_tinyllama_inference.py \
      --lazy-load \
      --dynamic-load \
      --batch-size $BATCH_SIZE \
      --split-layers $SPLIT_LAYERS \
      --init-method $INIT_METHOD \
      --model-dir $MODEL_DIR \
      --csv-has-header \
      --prompt-column prompt \
      --max-input-tokens $MAX_INPUT_TOKENS \
      $FORCE_DECODE_STEPS_ARG
fi
