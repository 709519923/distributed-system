# Distributed TinyLlama NCCL Inference

本项目用于 TinyLlama 两节点 / 三节点 NCCL 分布式推理实验。当前统一通过 `run.sh` 启动，实验参数集中写在 `run.sh` 顶部。

## 启动方式

三台机器分别进入项目目录后运行：

```bash
./run.sh 0
./run.sh 1
./run.sh 2
```

建议三台机器都放在各自的 `tmux` 窗口中运行，方便查看状态和日志。

## run.sh 配置

常用配置在 `run.sh` 顶部：

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
FORCE_DECODE_STEPS=${FORCE_DECODE_STEPS:-}
```

含义：

```text
WORLD_SIZE_VALUE  节点总数，当前常用 3
PREFILL_MODE      distributed 或 cloud-base
BATCH_SIZE        每个 batch 的 prompt 数量
SPLIT_LAYERS      默认 transformer 层切分点，例如 5,15
SCHEDULER_CSV     Rank 0 使用的调度文件，不存在时自动创建
INIT_METHOD       torch.distributed rendezvous 地址
COMPUTE_DEVICE    Rank 0 的计算设备，cuda 或 cpu
MODEL_DIR         TinyLlama 模型目录
INPUT_CSV         Rank 0 读取的输入 CSV
OUTPUT_CSV        Rank 0 写出的结果 CSV
MAX_INPUT_TOKENS  输入 prompt 最大 token 长度
FORCE_DECODE_STEPS 固定 decode forward 次数；空值表示按 EOS / max_new_tokens 自然停止
```

如果要固定 decode 阶段执行 128 步，在 `run.sh` 顶部设置：

```bash
FORCE_DECODE_STEPS=${FORCE_DECODE_STEPS:-128}
```

该参数对应命令行 `--force-decode-steps 128`。它会忽略 EOS，强制执行 128 次 prefill 之后的 decode forward。prefill 直接得到的 first token 不计入这 128 步。

## Scheduler

`scheduler.csv` 由 Rank 0 维护，Rank 1 / Rank 2 不需要传入调度文件。Rank 0 会在每个 batch 开始时广播当前 layer boundaries。

如果 `scheduler.csv` 不存在，Rank 0 会按 `SPLIT_LAYERS` 自动创建：

```text
batch,rank0,rank1,rank2
1,"[0,5)","[5,15)","[15,22)"
```

如果某个 batch 没有明确写入，scheduler 会沿用最近一个已知 batch 的层分配，不会回退到默认分配。

## PREFILL_MODE

`distributed`：

```text
Rank 0 计算前段层 KV cache
Rank 1 计算中段层 KV cache
Rank 2 计算后段层 KV cache
```

`cloud-base`：

```text
Rank 0 -> Rank 2: input_ids + attention_mask
Rank 2: 用完整模型计算全量 KV cache
Rank 2 -> Rank 0: 分发 Rank 0 所需 KV cache
Rank 2 -> Rank 1: 分发 Rank 1 所需 KV cache
Rank 2: 保留自己的 KV cache
```

`cloud-base` 当前要求：

```text
WORLD_SIZE_VALUE=3
--dynamic-load
Rank 2 能够加载完整 TinyLlama
```

## 数据流

Decode 阶段：

```text
hidden states: Rank 0 -> Rank 1 -> Rank 2
next token   : Rank 2 -> Rank 0
metrics/log  : Rank 2 -> Rank 1 -> Rank 0
```

非 transformer 层的处理方式：

```text
Rank 0: input embedding
最后一个 rank: final norm + lm_head
```

## Environment Simulation

通信环境不再通过 `run.sh` 参数设置，而是在 `environment.py` 中配置，并由 Rank 0 广播给其他 rank。

五条链路索引：

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

`Bandwidth` 单位是 MB/s，其中 `1 MB = 1024 * 1024 bytes`。`None` 表示该链路不限速。`time_comm_delay` 单位是 ms，表示固定单向通信时延。

## 日志

日志写入：

```text
logs/log_YYYY-MM-DD-HH-MM.txt
```

每个 batch 完成后会立刻写入 record 和 summary。summary 只统计当前 batch，不再跨 batch 累加 total。

record 主要字段：

```text
prefill_param_size_mb              当前 rank 持有的模型参数大小
kv_cache_size_mb_after_prefill     当前 rank 的 KV cache 大小

distributed parameter:
prefill_comp_time_ms               prefill 阶段本 rank 计算时间
prefill_transfer_time_ms           prefill 阶段本 rank 发出数据的传输时间，包含环境模拟时延
                                  只用于 PREFILL_MODE=distributed；cloud-base 下应为 0.00

Cloud-base parameter:
cloud_prefill_rank2_time_ms        cloud-base 中 Rank 2 完整模型 prefill 时间
kv_cache_send_time_ms              cloud-base 中 Rank 2 并行发送 KV cache 时间
kv_cache_recv_time_ms              cloud-base 中 Rank 0 / Rank 1 接收并重建 KV cache 时间

Common parameter:
decode_step_count                  decode 自回归 forward 次数
decode_comp_time_ms                decode 阶段本 rank 计算时间
decode_transfer_time_ms            decode 阶段本 rank 发出数据的传输时间，包含环境模拟时延
decode_time_per_token_ms           (decode_comp_time_ms + decode_transfer_time_ms) / decode_step_count
                                  只统计 prefill 之后继续自回归的 decode；cloud-base 的 first token 不计入
```

summary 主要包含：

```text
当前 batch 的 PREFILL_MODE
当前 batch 的 layer_allocation
当前 batch 的 Environment 带宽和通信时延
每个 rank 当前 batch 的 Distributed / Cloud-base 汇总时间
非当前 PREFILL_MODE 的 summary 栏目会写 not applicable
```

## 排查建议

1. 三台机器代码必须一致，尤其是 `environment.py`、`inference_loops.py`、`experiment_report.py`、`pipeline_comm.py`、`kv_cache_transfer.py`、`bandwidth_transfer.py`。
2. 三台机器的 `WORLD_SIZE_VALUE`、`SPLIT_LAYERS`、`INIT_METHOD` 必须一致。
3. `PREFILL_MODE`、`scheduler.csv` 和 `environment.py` 以 Rank 0 为准。
4. 如果 NCCL 报网络错误，优先检查 `NCCL_SOCKET_IFNAME` 是否是互通网卡。
5. 如果 `cloud-base` 显存过高，注意 Rank 2 会同时持有完整 prefill model 和自己的 decode 分区。
