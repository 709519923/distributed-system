#!/bin/bash
# NCCL 分布式推理环境变量配置
export RANK=1
export WORLD_SIZE=2
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
