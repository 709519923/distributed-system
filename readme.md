# Distributed TinyLlama NCCL Inference

本项目用于三节点 TinyLlama 分布式推理实验。当前运行入口统一使用 `run.sh`。

## 当前数据流

三节点切分方式由 `SPLIT_LAYERS` 控制，例如：

```bash
SPLIT_LAYERS=5,15
```

表示：

```text
Rank 0: layer [0,5)
Rank 1: layer [5,15)
Rank 2: layer [15,22)
```

当前推理数据流：

```text
hidden states: Rank 0 -> Rank 1 -> Rank 2
next token   : Rank 2 -> Rank 0
metrics/log  : Rank 2 -> Rank 1 -> Rank 0
```

也就是说，Rank 1 只负责中间层 hidden states 转发，不再转发 Rank 2 生成的 token。

## 修改实验参数

直接编辑 `run.sh` 顶部配置区：

```bash
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
```

常用配置：

```text
BATCH_SIZE       每个 batch 的 prompt 数量
PREFILL_MODE     KV cache prefill 模式：distributed 或 cloud-base
SPLIT_LAYERS     三节点 layer 切分点
INIT_METHOD      master rendezvous 地址和端口
COMPUTE_DEVICE   仅 Rank 0 使用，cuda 表示 GPU compute，cpu 表示 CPU compute + CUDA/NCCL communication
MODEL_DIR        TinyLlama 模型目录
INPUT_CSV        Rank 0 读取的输入 CSV
OUTPUT_CSV       Rank 0 写出的结果 CSV
MAX_INPUT_TOKENS 输入 prompt 最大 token 长度
```

## 启动方式

三台机器分别进入项目目录后运行：

```bash
./run.sh 0
```

```bash
./run.sh 1
```

```bash
./run.sh 2
```

建议先启动 Rank 0，再启动 Rank 1 和 Rank 2。每台机器可以放在自己的 `tmux` 窗口中运行，方便回看日志。

## KV Cache Prefill 模式

当前支持两种 `PREFILL_MODE`：

```text
distributed
cloud-base
```

`distributed` 是默认模式，也是当前稳定模式：

```text
Rank 0 计算自己的前段 KV cache
Rank 1 计算自己的中段 KV cache
Rank 2 计算自己的后段 KV cache
```

`cloud-base` 是新增实验模式：

```text
Rank 0 -> Rank 2: input_ids + attention_mask
Rank 2: 使用完整模型计算整模型 KV cache
Rank 2 -> Rank 0: layer [0, split0) KV cache
Rank 2 -> Rank 1: layer [split0, split1) KV cache
Rank 2: 保留 layer [split1, end) KV cache
```

`cloud-base` 下后续 decode 仍然保持：

```text
hidden states: Rank 0 -> Rank 1 -> Rank 2
next token   : Rank 2 -> Rank 0
```

启用方式是在 Rank 0 / Rank 1 / Rank 2 的 `run.sh` 顶部都设置：

```bash
PREFILL_MODE=${PREFILL_MODE:-cloud-base}
```

第一版 `cloud-base` 仅支持：

```text
WORLD_SIZE_VALUE=3
--dynamic-load
Rank 2 能加载完整 TinyLlama
```

## Rank 0 CPU Compute

如果希望 Rank 0 使用 CPU 计算前段模型，同时继续使用 CUDA/NCCL 做跨节点通信，把 `run.sh` 顶部改成：

```bash
COMPUTE_DEVICE=${COMPUTE_DEVICE:-cpu}
```

或者临时启动 Rank 0：

```bash
COMPUTE_DEVICE=cpu ./run.sh 0
```

Rank 1 和 Rank 2 不需要设置 `COMPUTE_DEVICE`，仍然使用 GPU。

CPU compute 模式仍然要求 Rank 0 有可用 CUDA 设备，因为 hidden states 和 token 的跨节点通信仍然走 NCCL。

## 输入 CSV

输入文件只需要包含 prompt 列：

```csv
prompt
Please introduce TinyLlama in one short paragraph.
Write a short checklist for debugging NCCL communication.
```

`run.sh` 当前默认读取：

```text
./dataset/input10.csv
```

如需更换数据集，修改 `INPUT_CSV`。

## 输出文件

推理结果默认写入：

```text
outputs_kv.csv
```

实验日志写入：

```text
logs/log_YYYY-MM-DD-HH-MM.txt
```

日志中每个 batch 会写入每个 rank 的记录，并在 batch 完成后追加累计汇总：

```text
--- summary after batch N ---
```

重点字段：

```text
prefill_time_ms                 当前 rank 的本地 prefill forward 时间
prefill_time_total_ms           当前 batch 的 prefill 统计时间
decode_step_count               decode 自回归 forward 次数
decode_time_total_ms            当前 rank decode 总时间
decode_time_per_token_ms        当前 rank 平均每次 decode forward 时间
inference_compute_total_ms      prefill + decode 的本地计算总时间
cloud_prefill_rank2_time_ms     cloud-base 下 Rank 2 完整模型 prefill 时间
kv_cache_send_time_ms           cloud-base 下 Rank 2 并行发送 KV cache 的墙钟时间
kv_cache_recv_time_ms           cloud-base 下 Rank 0 / Rank 1 接收 KV cache 的时间
total_prefill_time_ms           已完成 batch 的累计 prefill 时间
total_decode_time_ms            已完成 batch 的累计 decode 时间
total_inference_compute_time_ms 已完成 batch 的累计计算时间
total_decode_step_count         已完成 batch 的累计 decode 次数
```

## 注意事项

1. 三台机器代码必须保持一致，尤其是 `experiment_report.py` 里的 metric 字段顺序。
2. 三台机器的 `WORLD_SIZE_VALUE`、`SPLIT_LAYERS`、`INIT_METHOD` 必须一致。
3. Rank 0 需要能直接接收 Rank 2 的 NCCL token 返回连接。
4. 如果修改 `INIT_METHOD` 端口，三台机器必须同时修改。
5. 如果 NCCL 报网络错误，优先检查 `NCCL_SOCKET_IFNAME` 是否是互通网卡。
6. `cloud-base` 会让 Rank 2 同时持有完整 prefill model 和自己的 decode 分区，显存压力会明显高于 `distributed`。
