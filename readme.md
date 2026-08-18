# TinyLlama Lipschitz 可行性验证

本分支用于穷举 TinyLlama 三节点的全部层切分，验证 arm 距离与推理性能之间是否存在可利用的 Lipschitz 关系。

## 实验配置

实验参数统一在 `run.sh` 顶部修改，默认配置为：

```bash
WORLD_SIZE_VALUE=3
PREFILL_MODE=distributed
BATCH_SIZE=1
SPLIT_LAYERS=5,15
SCHEDULER_CSV=lipschitz_validation_scheduler.csv
BANDIT_POLICY=lipschitz_validation
MODEL_DIR=/home/dingcong/models/TinyLlama
INPUT_CSV=./dataset/lipschitz_validation_prompts.csv
OUTPUT_CSV=lipschitz_validation_outputs.csv
FORCE_DECODE_STEPS=128
```

`lipschitz_validation` 只支持三节点 TinyLlama。22 个 decoder layer 的所有合法 arm 为：

```text
0 < p1 < p2 < 22
rank0=[0,p1)  rank1=[p1,p2)  rank2=[p2,22)
```

总 arm 数为：

$$
\binom{21}{2}=210
$$

## 启动

三台机器同步代码后分别执行：

```bash
./run.sh 0
./run.sh 1
./run.sh 2
```

Rank 0 读取 CSV、生成 210 个 arm 的执行计划并广播切分点；Rank 1 和 Rank 2 不读取 scheduler 文件，只接收 Rank 0 的广播。

## 输入数据

CSV 只需要 `prompt` 表头：

```csv
prompt
Your first prompt
Your second prompt
```

程序支持任意数量的 prompt。设 prompt 数为 $N$，`BATCH_SIZE` 为 $B$，则逻辑 batch 数和真实三节点执行次数分别为：

$$
D=\left\lceil\frac{N}{B}\right\rceil,\qquad N_{exec}=210D
$$

建议使用 `BATCH_SIZE=1`，这样每个逻辑 batch 对应一个 prompt，得到的是 prompt 级 ground truth。`BATCH_SIZE>1` 时，排名表示整个 tensor batch 的性能。

## 输出文件

程序边运行边写文件：

```text
lipschitz_validation_outputs.csv
bandit_logs/lipschitz_validation_raw_YYYY-MM-DD-HH-MM-SS.csv
bandit_logs/lipschitz_validation_ranked_YYYY-MM-DD-HH-MM-SS.csv
logs/log_YYYY-MM-DD-HH-MM.txt
```

- `lipschitz_validation_outputs.csv`：每个 arm 生成的文本，并带逻辑 batch、执行 batch 和 arm 编号。
- `raw`：每个 arm 完成后立即追加并 `fsync`，中途失败时已完成数据仍然保留。
- `ranked`：一个逻辑 batch 的 210 个 arm 全部完成后，一次写入完整排名。
- `logs`：原有逐 rank 性能记录，每个执行 batch 完成后立即写入。

`logical_batch` 表示同一组 prompt；`execution_batch` 是现有分布式通信协议使用的唯一连续编号：

$$
execution\_batch=(logical\_batch-1)\times210+execution\_order
$$

## 排名指标

先按当前 prefill 模式计算每个 rank 的时间 $T_r$，再取最慢节点：

$$
T_{arm}=\max(T_0,T_1,T_2)
$$

固定 decode step 后，单位 step 成本和便于观察的分数为：

$$
C_{arm}=\frac{T_{arm}}{\max(1,N_{decode})},\qquad
S_{arm}=\frac{1}{1+C_{arm}/100}
$$

排名按 $C_{arm}$ 从小到大。该模式不更新 reward，不执行 UCB 探索，也不淘汰 arm。

## 注意事项

1. `FORCE_DECODE_STEPS` 必须设置为正整数，默认 128，保证所有 arm 的 decode 工作量一致。
2. 同一逻辑 batch 的 210 个 arm 使用同一个 Environment batch 状态。
3. arm 顺序采用固定随机种子并在不同逻辑 batch 间轮转，减少固定执行位置带来的系统偏差。
4. 程序不暗中增加预热执行，因此总次数严格为 $210D$。首次 CUDA/模型调用的冷启动影响应在离线分析时单独标记。
5. 三台机器必须同步 `scheduler.py`、`inference_loops.py`、`config.py` 和 `lipschitz_validation_experiment.py`。
