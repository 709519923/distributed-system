# Distributed TinyLlama NCCL Inference

本项目用于 TinyLlama 三节点 NCCL 分布式推理实验。当前统一通过 `run.sh` 启动。

## 运行入口

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

建议三台机器都放在各自的 `tmux` 窗口中运行，方便回看日志。

## run.sh 配置

实验参数集中写在 `run.sh` 顶部：

```bash
WORLD_SIZE_VALUE=${WORLD_SIZE_VALUE:-3}
PREFILL_MODE=${PREFILL_MODE:-distributed}
BATCH_SIZE=${BATCH_SIZE:-1}
SPLIT_LAYERS=${SPLIT_LAYERS:-5,15}
SCHEDULER_CSV=${SCHEDULER_CSV:-scheduler.csv}
INIT_METHOD=${INIT_METHOD:-tcp://10.50.1.228:29510}
COMPUTE_DEVICE=${COMPUTE_DEVICE:-cuda}
MODEL_DIR=${MODEL_DIR:-/home/dingcong/models/TinyLlama}
INPUT_CSV=${INPUT_CSV:-./dataset/input10.csv}
OUTPUT_CSV=${OUTPUT_CSV:-outputs_kv.csv}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-1000}
```

常用含义：

```text
BATCH_SIZE       每个 batch 的 prompt 数量
PREFILL_MODE     KV cache prefill 模式：distributed 或 cloud-base
SPLIT_LAYERS     默认 layer 切分点，例如 5,15
SCHEDULER_CSV    Rank 0 调度文件；不存在时按 SPLIT_LAYERS 自动创建
INIT_METHOD      torch.distributed rendezvous 地址和端口
COMPUTE_DEVICE   只影响 Rank 0：cuda 或 cpu
MODEL_DIR        TinyLlama 模型目录
INPUT_CSV        Rank 0 读取的输入 CSV
OUTPUT_CSV       Rank 0 写出的结果 CSV
MAX_INPUT_TOKENS 输入 prompt 最大 token 长度
```

修改参数时，直接改 `run.sh` 顶部配置块。三台机器的启动命令仍然固定为 `./run.sh 0`、`./run.sh 1`、`./run.sh 2`。

## Scheduler

`scheduler.csv` 现在是默认调度文件，由 Rank 0 维护。Rank 1 / Rank 2 不需要传入 `--allocation-csv`，它们只接收 Rank 0 广播的 layer boundaries。

如果 `scheduler.csv` 不存在，Rank 0 会在第一个 batch 按 `SPLIT_LAYERS` 自动创建。例如：

```text
batch,rank0,rank1,rank2
1,"[0,5)","[5,15)","[15,22)"
```

如果某个 batch 没有明确写在 `scheduler.csv` 中，scheduler 会沿用最近一个已知 batch 的 layer 分配，不会回退到默认分配。

`scheduler.py` 中已经预留 `reallocate_layer()`。当前版本该方法只返回最新分配，不主动改层分配；后续可以根据 Rank 0 / Rank 1 / Rank 2 的统计数据在这里实现动态重分配策略。

## Layer 分配

例如：

```bash
SPLIT_LAYERS=5,15
```

表示：

```text
Rank 0: layer [0,5)
Rank 1: layer [5,15)
Rank 2: layer [15,22)
```

TinyLlama 的非 transformer 层仍然按当前程序逻辑处理：Rank 0 负责输入 embedding，末尾 rank 负责 final norm / lm head。

## 数据流

Decode 阶段的数据流为：

```text
hidden states: Rank 0 -> Rank 1 -> Rank 2
next token   : Rank 2 -> Rank 0
metrics/log  : Rank 2 -> Rank 1 -> Rank 0
```

Rank 1 只处理中间层 hidden states，不再转发 Rank 2 生成的 token。

## PREFILL_MODE

`PREFILL_MODE` 只由 Rank 0 设置并主导。Rank 1 / Rank 2 不需要在命令行里传 `--prefill-mode`，程序启动后会通过 NCCL 从 Rank 0 接收最终模式。

`distributed` 表示每个 rank 自己计算自己分配层的 KV cache。

`cloud-base` 表示：

```text
Rank 0 -> Rank 2: input_ids + attention_mask
Rank 2: 使用完整模型计算整模型 KV cache
Rank 2 -> Rank 0: layer [0, split0) KV cache
Rank 2 -> Rank 1: layer [split0, split1) KV cache
Rank 2: 保留 layer [split1, end) KV cache
```

`cloud-base` 要求：

```text
WORLD_SIZE_VALUE=3
--dynamic-load
Rank 2 能加载完整 TinyLlama
```

## Environment Simulation

通信环境不再通过 `run.sh` 设置。现在统一在 `environment.py` 中配置，并由 Rank 0 广播给 Rank 1 / Rank 2。

`environment.py` 维护 5 条真实通信链路：

```text
Bandwidth[0], time_comm_delay[0] = Rank 0 -> Rank 1
Bandwidth[1], time_comm_delay[1] = Rank 1 -> Rank 2
Bandwidth[2], time_comm_delay[2] = Rank 2 -> Rank 0
Bandwidth[3], time_comm_delay[3] = Rank 0 -> Rank 2
Bandwidth[4], time_comm_delay[4] = Rank 2 -> Rank 1
```

默认配置：

```python
DEFAULT_BANDWIDTH = [None, None, None, None, None]
DEFAULT_TIME_COMM_DELAY = [0.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_SCHEDULE = {}
```

含义：

```text
Bandwidth: MB/s；None 表示该链路不限速
time_comm_delay: ms；表示固定单向通信时延
```

模拟公式：

```text
target_seconds = payload_bytes / (Bandwidth[i] * 1024 * 1024)
delay_seconds = time_comm_delay[i] / 1000
expected_seconds = target_seconds + delay_seconds
extra_sleep = max(0, expected_seconds - real_elapsed_seconds)
```

实际代码只补足真实 NCCL 通信没有覆盖的部分，避免真实通信已经很慢时被重复惩罚。

预留 batch 后变化：

```python
DEFAULT_SCHEDULE = {
    10: {
        "Bandwidth": [100, 80, 120, 60, 70],
        "time_comm_delay": [1.0, 1.5, 2.0, 5.0, 4.0],
    }
}
```

表示从 batch 10 开始切换到新的网络环境。动态模式中，每个 batch 开始时 Rank 0 会应用当前 batch 的环境配置并广播。

## Rank 0 CPU Compute

如果希望 Rank 0 使用 CPU 计算前段模型，同时继续用 CUDA/NCCL 通信，在 `run.sh` 顶部设置：

```bash
COMPUTE_DEVICE=${COMPUTE_DEVICE:-cpu}
```

Rank 1 和 Rank 2 不需要设置 `COMPUTE_DEVICE`，仍然使用 GPU。CPU compute 模式仍要求 Rank 0 有可用 CUDA 设备，因为跨节点通信仍走 NCCL。

## 输入输出

输入 CSV 保持原格式，一行一条 prompt。默认读取：

```text
./dataset/input10.csv
```

Rank 0 写出：

```text
outputs_kv.csv
```

实验日志写入：

```text
logs/log_YYYY-MM-DD-HH-MM.txt
```

日志会在每个 batch 完成后立即写入，避免长任务中断后丢失已完成 batch 的统计。

## 日志字段

常用字段：

```text
prefill_time_ms                 当前 rank 的本地 prefill forward 时间
prefill_time_total_ms           当前 batch 的 prefill 统计时间
decode_step_count               decode 自回归 forward 次数
decode_time_total_ms            当前 rank decode 总时间
decode_time_per_token_ms        当前 rank 平均每次 decode forward 时间
inference_compute_total_ms      prefill + decode 的本地计算总时间
cloud_prefill_rank2_time_ms     cloud-base 中 Rank 2 完整模型 prefill 时间
kv_cache_send_time_ms           cloud-base 中 Rank 2 并行发送 KV cache 的墙钟时间
kv_cache_recv_time_ms           cloud-base 中 Rank 0 / Rank 1 接收并重建 KV cache 的时间
```

## 排查建议

1. 三台机器的代码必须完全一致，尤其是 `config.py`、`distributed_env.py`、`environment.py`、`inference_loops.py`、`pipeline_comm.py`、`kv_cache_transfer.py`、`bandwidth_transfer.py`、`scheduler.py`、`experiment_report.py`。
2. 三台机器的 `WORLD_SIZE_VALUE`、`SPLIT_LAYERS`、`INIT_METHOD` 必须一致。
3. 只有 Rank 0 需要设置 `PREFILL_MODE`、`SCHEDULER_CSV` 和 `environment.py` 中的环境参数；Rank 1 / Rank 2 应以广播结果为准。
4. 如果 NCCL 报网络错误，优先检查 `NCCL_SOCKET_IFNAME` 是否是互通网卡。
5. 如果 scheduler 行为不符合预期，先检查 Rank 0 本地的 `scheduler.csv` 是否存在、表头是否为 `batch,rank0,rank1,rank2`、对应 batch 是否有明确分配。
6. `cloud-base` 会让 Rank 2 同时持有完整 prefill model 和自己的 decode 分区，显存压力会高于 `distributed`。
