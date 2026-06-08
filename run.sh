#!/bin/bash

BATCH_SIZE=1
SPLIT_LAYERS=5,15
INIT_METHOD=tcp://10.50.1.228:29510
COMPUTE_DEVICE=${COMPUTE_DEVICE:-cuda}
RANK_ARG=$1

if [ "$RANK_ARG" = "0" ]; then
    export WORLD_SIZE=3
    export RANK=0
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_DEBUG=INFO

    python distributed_tinyllama_inference.py \
      --lazy-load \
      --dynamic-load \
      --batch-size $BATCH_SIZE \
      --split-layers $SPLIT_LAYERS \
      --compute-device $COMPUTE_DEVICE \
      --init-method $INIT_METHOD \
      --model-dir /home/dingcong/models/TinyLlama \
      --input-csv ./dataset/input10.csv \
      --output-csv outputs_kv.csv \
      --csv-has-header \
      --prompt-column prompt \
      --max-input-tokens 1000
fi

if [ "$RANK_ARG" = "1" ]; then
    export WORLD_SIZE=3
    export RANK=1
    export NCCL_SOCKET_IFNAME=enp6s18
    export NCCL_DEBUG=INFO

    python distributed_tinyllama_inference.py \
      --lazy-load \
      --dynamic-load \
      --batch-size $BATCH_SIZE \
      --split-layers $SPLIT_LAYERS \
      --init-method $INIT_METHOD \
      --model-dir /home/dingcong/models/TinyLlama \
      --csv-has-header \
      --prompt-column prompt \
      --max-input-tokens 1000
fi

if [ "$RANK_ARG" = "2" ]; then
    export WORLD_SIZE=3
    export RANK=2
    export NCCL_SOCKET_IFNAME=enp6s18
    export NCCL_DEBUG=INFO

    python distributed_tinyllama_inference.py \
      --lazy-load \
      --dynamic-load \
      --batch-size $BATCH_SIZE \
      --split-layers $SPLIT_LAYERS \
      --init-method $INIT_METHOD \
      --model-dir /home/dingcong/models/TinyLlama \
      --csv-has-header \
      --prompt-column prompt \
      --max-input-tokens 1000
fi
