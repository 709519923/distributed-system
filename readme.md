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
BANDIT_POLICY=${BANDIT_POLICY:-ucb}
EXPERIMENT_SCENARIO=${EXPERIMENT_SCENARIO:-D}
SCENARIO_LOWER=${EXPERIMENT_SCENARIO,,}
RUN_TAG=${RUN_TAG:-${EXPERIMENT_SCENARIO}}
SCHEDULER_CSV=${SCHEDULER_CSV:-scheduler_${BANDIT_POLICY}_${RUN_TAG}.csv}
INIT_METHOD=${INIT_METHOD:-tcp://10.50.1.130:29510}
COMPUTE_DEVICE=${COMPUTE_DEVICE:-cuda}
MODEL_DIR=${MODEL_DIR:-/home/dingcong/models/TinyLlama}
INPUT_CSV=${INPUT_CSV:-./dataset/single_scenario_${SCENARIO_LOWER}_500.csv}
OUTPUT_CSV=${OUTPUT_CSV:-outputs_${BANDIT_POLICY}_${RUN_TAG}.csv}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-1800}
FORCE_DECODE_STEPS=${FORCE_DECODE_STEPS:-}
```

含义：

```text
WORLD_SIZE_VALUE  节点总数，当前常用 3
PREFILL_MODE      distributed 或 cloud-base
BATCH_SIZE        每个 batch 的 prompt 数量
SPLIT_LAYERS      默认 transformer 层切分点，例如 5,15
SCHEDULER_CSV     Rank 0 使用的调度文件，不存在时自动创建
BANDIT_POLICY     ucb、lipschitz、epsilon_greedy、thompson_sampling 等调度策略
INIT_METHOD       torch.distributed rendezvous 地址
COMPUTE_DEVICE    Rank 0 的计算设备，cuda 或 cpu
MODEL_DIR         TinyLlama 模型目录
INPUT_CSV         Rank 0 读取的输入 CSV
OUTPUT_CSV        Rank 0 写出的结果 CSV
MAX_INPUT_TOKENS  输入 prompt 最大 token 长度
EXPERIMENT_SCENARIO 单场景 bandit 实验；可选 A、B、C、D、E、F
FORCE_DECODE_STEPS 固定 decode forward 次数；空值表示按 EOS / max_new_tokens 自然停止
```

如果要固定 decode 阶段执行 128 步，在 `run.sh` 顶部设置：

```bash
FORCE_DECODE_STEPS=${FORCE_DECODE_STEPS:-128}
```

该参数对应命令行 `--force-decode-steps 128`。它会忽略 EOS，强制执行 128 次 prefill 之后的 decode forward。prefill 直接得到的 first token 不计入这 128 步。

## A-F 单场景四算法对比

当前分支的 `run.sh` 默认运行 D 场景，但可通过 `EXPERIMENT_SCENARIO=A` 到
`EXPERIMENT_SCENARIO=F` 切换。CSV 只需包含 `prompt` 列，不限制数据行数；
`BATCH_SIZE=1` 时每一行自然形成一个 batch，输入多少行就运行多少个 batch。
默认网络环境在 `environment.py` 中固定为：

```text
Rank 0 -> Rank 1: 10 ms
Rank 1 -> Rank 2: 30 ms
Rank 2 -> Rank 0: 40 ms
其他两条链路: 0 ms
```

四组实验分别把 `BANDIT_POLICY` 设为：

```bash
BANDIT_POLICY=ucb
BANDIT_POLICY=lipschitz
BANDIT_POLICY=epsilon_greedy
BANDIT_POLICY=thompson_sampling
```

每组都在三台机器上运行 `./run.sh 0`、`./run.sh 1`、`./run.sh 2`。Rank 0
会根据 policy 自动使用互不覆盖的 scheduler/output 文件名。若数据文件不在
`./dataset/`，可显式传入 `INPUT_CSV=/path/to/data.csv`。

例如切换到 E 场景时，Rank 0 可以运行：

```bash
EXPERIMENT_SCENARIO=E BANDIT_POLICY=lipschitz ./run.sh 0
```

未显式设置 `INPUT_CSV` 时，脚本会随场景自动选择
`single_scenario_a_500.csv` 到 `single_scenario_f_500.csv`。使用其他行数的
数据集时直接设置 `INPUT_CSV` 即可。

四个策略共用 195 个候选 arm、seed 42、首个完成 batch 全局 warmup、逐 batch
reward 和计时口径。关键参数为：

| Policy | 参数/规则 |
|---|---|
| `ucb` | 探索权重 `0.01`，未观测 arm 优先，之后使用经典 UCB1 score |
| `lipschitz` | 探索权重 `0.05`，`q1=q2=0.5` 初始化，使用在线距离与置信界 |
| `epsilon_greedy` | `epsilon=0.05`，每个 arm 先观测一次，再做探索/利用 |
| `thompson_sampling` | 每个 arm 使用 `Beta(1,1)` 先验，连续 reward 作分数计数更新 |

各场景的输入检查范围和固定输出为：

| 场景 | 输入 token 范围 | 输出 token IDs |
|---|---:|---:|
| A | 800–1500 | 90 |
| B | 10–150 | 400 |
| C | 200–600 | 256 |
| D | 1500–1800 | 32 |
| E | 10–64 | 768 |
| F | 800–1100 | 512 |

prefill first token 计入目标总数，所以实际 `decode_step_count=target_output_tokens-1`。
scenario 只控制输出长度与日志标签，不参与四个非 contextual policy 选臂。该模式不能与
`FORCE_DECODE_STEPS` 同时使用；decode 参数仍由实验者使用现有配置手动调整。

每次运行新增：

```text
bandit_logs/batch_metrics_{policy}_{scenario}_{timestamp}.csv
bandit_logs/run_summary_{policy}_{scenario}_{timestamp}.csv
```

前者每个 batch 一行，记录 `policy`、策略参数、`used_arm`、`next_arm`、统一
reward、累计 reward、`scheduler_select_ms`、`scheduler_update_ms`、
`scheduler_algorithm_ms`、三条链路的实际 delay、推理墙钟时间和切层时间。
算法计时不包含 CSV/文本日志写盘；Lipschitz 为详细审计生成置信界快照的时间
单独记录在 `scheduler_audit_ms`，也不混入算法时间。后者记录整次运行汇总，并额外给出 scheduler 算法耗时的
mean/p50/p95/max 以及 `algorithm_overhead_percent`。

四组都完成后运行：

```bash
python extract_bandit_results.py \
  --log-dir bandit_logs \
  --scenario D
```

脚本根据配套 `run_summary` 的 `status=complete` 识别完整运行，并检查四个
policy 的 batch 数相同、batch 号从 1 连续排列。输出：

```text
bandit_logs/comparison_batches_D.csv   # 4 × N 行逐 batch 数据
bandit_logs/comparison_summary_D.csv   # 4 行算法汇总
```

## Contextual Controlled Policy

`contextual_controlled` 专用于 `contextual_bandit_test_tinyllama_labeled.csv` 的受控学习/评估实验。Rank 0 的 `run.sh` 配置为：

```bash
BANDIT_POLICY=${BANDIT_POLICY:-contextual_controlled}
BATCH_SIZE=${BATCH_SIZE:-1}
INPUT_CSV=${INPUT_CSV:-./dataset/contextual_bandit_test_tinyllama_labeled.csv}
FORCE_DECODE_STEPS=${FORCE_DECODE_STEPS:-}
```

输入 CSV 必须包含：

```text
prompt,scenario,request_type,phase
```

该 policy 要求正好 900 行和 20 个互不重复的实际有效 arm。默认 split 必须已经包含在 `CANDIDATE_ARMS` 中，避免 Scheduler 额外插入第 21 个 arm。

前 600 个 batch 是 learning 阶段，数据按 A/B/C 循环。每个 arm 在每种 request type 下学习 10 次：

```text
A / long_input_short_output    -> 强制保存 90 个 token ID
B / short_input_long_output    -> 强制保存 400 个 token ID
C / medium_input_medium_output -> 强制保存 256 个 token ID
```

Label 只控制 learning 阶段的输出 token 数，Context 特征仍由 prompt 的实际 token 长度生成。强制输出时忽略 EOS；prefill first token 计入目标总数，因此 `decode_step_count=target_output_tokens-1`。

Batch 601-900 是 evaluation 阶段：

```text
不读取 scenario/request_type label 参与决策
根据 prompt token 长度构造 Context
使用前 600 批学到的 LinUCB 模型选臂
冻结每个 arm 的 A/b，不再更新模型
按 EOS 自然停止，最大输出仍为 --max-new-tokens（默认 512）
```

`contextual_controlled` 不能与 `--force-decode-steps` 同时使用，并且必须启用 `--allocation-csv`、`--csv-has-header` 和 `--batch-size 1`。

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
DEFAULT_TIME_COMM_DELAY = [10.0, 30.0, 40.0, 0.0, 0.0]
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
