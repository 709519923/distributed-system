#!/bin/bash

# Edit this block for each experiment. Rank 0 is the only rank that uses
# COMPUTE_DEVICE; set it to "cpu" for CPU compute + CUDA/NCCL communication.
WORLD_SIZE_VALUE=${WORLD_SIZE_VALUE:-3}
PREFILL_MODE=${PREFILL_MODE:-distributed}
BATCH_SIZE=${BATCH_SIZE:-1}
SPLIT_LAYERS=${SPLIT_LAYERS:-5,15}
SCHEDULER_CSV=${SCHEDULER_CSV:-scheduler.csv}
BANDIT_POLICY=${BANDIT_POLICY:-contextual_controlled}
INIT_METHOD=${INIT_METHOD:-tcp://10.50.1.130:29510}
COMPUTE_DEVICE=${COMPUTE_DEVICE:-cuda}
MODEL_DIR=${MODEL_DIR:-/home/dingcong/models/TinyLlama}
INPUT_CSV=${INPUT_CSV:-./dataset/contextual_bandit_test_tinyllama.csv}
CONTEXT_MANIFEST=${CONTEXT_MANIFEST:-}
OUTPUT_CSV=${OUTPUT_CSV:-outputs_kv.csv}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-1800}
EXPERIMENT_SCENARIO=${EXPERIMENT_SCENARIO:-}
FORCE_DECODE_STEPS=${FORCE_DECODE_STEPS:-}
TOP_K_ARMS=${TOP_K_ARMS:-5}

EXPERIMENT_SCENARIO_ARG=""
if [ -n "$EXPERIMENT_SCENARIO" ]; then
    EXPERIMENT_SCENARIO_ARG="--experiment-scenario $EXPERIMENT_SCENARIO"
fi

FORCE_DECODE_STEPS_ARG=""
if [ -n "$FORCE_DECODE_STEPS" ]; then
    FORCE_DECODE_STEPS_ARG="--force-decode-steps $FORCE_DECODE_STEPS"
fi

CONTEXT_MANIFEST_ARG=""
if [ -n "$CONTEXT_MANIFEST" ]; then
    CONTEXT_MANIFEST_ARG="--context-manifest $CONTEXT_MANIFEST"
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

echo "[run.sh] RANK=$RANK_ARG WORLD_SIZE=$WORLD_SIZE_VALUE PREFILL_MODE=$PREFILL_MODE_TEXT BATCH_SIZE=$BATCH_SIZE SPLIT_LAYERS=$SPLIT_LAYERS SCHEDULER_CSV=$SCHEDULER_CSV BANDIT_POLICY=$BANDIT_POLICY CONTEXT_MANIFEST=${CONTEXT_MANIFEST:-off} TOP_K_ARMS=$TOP_K_ARMS EXPERIMENT_SCENARIO=${EXPERIMENT_SCENARIO:-off} INIT_METHOD=$INIT_METHOD COMPUTE_DEVICE=$COMPUTE_DEVICE FORCE_DECODE_STEPS=${FORCE_DECODE_STEPS:-off}"

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
      --top-k-arms $TOP_K_ARMS \
      --csv-has-header \
      --prompt-column prompt \
      --max-input-tokens $MAX_INPUT_TOKENS \
      $CONTEXT_MANIFEST_ARG \
      $EXPERIMENT_SCENARIO_ARG \
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
