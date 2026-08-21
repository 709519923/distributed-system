# Version Log

## 2026-08-21

### Ground Truth D/E/F 差异化场景与数据集

#### 设计动机

对本机已有的 6 份 ground-truth 结果进行汇总后，每种旧场景覆盖 300 个逻辑
batch。A 的 best/second-best 相对 cost 差距中位数只有 0.63%，并且 300 次中有
299 次由 Rank 0 构成瓶颈；`(1,11)`、`(1,13)`、`(1,15)`、`(1,17)` 的平均
cost 分别为 12.552、12.538、12.537、12.552 ms/token。由于这些 arm 给 Rank 0
分配的层数完全相同，当 Rank 0 已经决定流水线吞吐时，改变第二个切分点不会改变
`max(rank_time)`，因此它们形成真实的近似等价平台，而不只是样本量不足。

B 和 C 已开始集中到 `(1,13)`、`(1,15)`，但 best/second-best 差距中位数仍只有
0.86% 和 0.54%。已有 900 个逻辑 batch 中 `(3,x)` 没有成为过 best arm，说明
同一硬件条件下第一个切分点的粗粒度候选也是当前可分性的限制之一。

新实验先保持三台机器、计时定义和 `CANDIDATE_ARMS` 不变，只放大请求 workload
的差异，以便把场景影响与 arm-list 影响分开：

| Scenario | Request type | Input token | 固定输出 token | 目的 |
|---|---|---:|---:|---|
| D | `extreme_prefill` | 1500-1800 | 32 | 极端 prefill 对照，验证 `(1,x)` 平台是否由 Rank 0 瓶颈造成 |
| E | `extreme_decode` | 10-64 | 768 | 放大 decode 占比，观察 Rank 1/2 平衡点是否更明确 |
| F | `long_context_decode` | 800-1100 | 512 | 同时增加 KV 上下文与 decode 计算，观察第二切分点是否迁移 |

D 不预设必须得到唯一 best arm；如果它仍产生宽平台，同时 E/F 的 top-arm 更集中，
这本身就是“固定 Rank 0 瓶颈”与“可由第二切分点平衡的 decode workload”之间的有效
区分。后续分析除唯一排名外，还应报告 1%/3% epsilon-optimal arm 集合。

#### 上下文预算

运行模型 TinyLlama 的 `max_position_embeddings=2048`。数据准备和运行时校验都使用
同一个保守约束：

$$
L_{in}+L_{out}+1\le 2048,
$$

其中额外的 1 个 token 给 BOS/特殊 token 留余量。三个场景最坏预算分别为
D=1833、E=833、F=1613，均不依赖截断才能运行。`MAX_INPUT_TOKENS=2000` 也不会
截断 D/F 的合法 prompt。

#### 单场景数据构建

更新 `C:/Users/smbu/Desktop/lab7/prepare_dataset/build_single_scenario_dataset.py`：

- 保留 A/B/C collector，同时新增 D/E/F，并改为可重复调用的命令行参数
  `--scenario`、`--target-rows`、`--output-csv`、`--tokenizer-dir`。
- 默认 tokenizer 改为运行时一致的 `/home/dingcong/models/TinyLlama`；所有筛选
  长度均使用 `add_special_tokens=False`，与 scheduler/ground-truth 的上下文长度
  口径一致。
- D 从 CNN/DailyMail 中筛选 1500-1800 token 的文章，使用极短摘要请求。
- E 从 WritingPrompts 中筛选 10-64 token 的写作 prompt，并要求参考 story 至少
  768 token，避免用短答案语义模拟长输出。
- F 使用 WritingPrompts 的 story 构造续写任务。脚本用 tokenizer 对 story 前缀
  做二分截取，使最终 prompt 严格落在 800-1100 token，并保留至少 512 个参考
  continuation token。目标输入长度使用与 301 个可选长度互质的步长 73 遍历，避免
  500 条样本集中在区间下半部。
- collector 对 prompt 去重；输出仍是只有 `prompt` 一列的 CSV，方便复用现有读取
  逻辑。

生成文件位于 `C:/Users/smbu/Desktop/lab7/dataset`：

- `single_scenario_d_500.csv`
- `single_scenario_e_500.csv`
- `single_scenario_f_500.csv`

使用 TinyLlama tokenizer 对写出的 1500 条 prompt 再次独立校验，结果为：

| Scenario | 行数 | 实测 min | Q1 | median | mean | Q3 | max | 最小上下文余量 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D | 500 | 1500 | 1547.0 | 1609.0 | 1624.54 | 1695.2 | 1799 | 216 |
| E | 500 | 12 | 29.0 | 39.0 | 38.87 | 49.0 | 64 | 1215 |
| F | 500 | 800 | 874.0 | 948.5 | 947.43 | 1020.2 | 1100 | 435 |

每个场景内部无重复 prompt，D/E/F 之间也无交叉重复。

#### 交错集、切片与 HTML 报告

重构 `C:/Users/smbu/Desktop/lab7/dataset/prepare_ground_truth_dataset.py`。脚本支持
`--scenario-set ABC|DEF`，会加载实际 TinyLlama tokenizer，重新校验行数、唯一性、
输入范围、输出标签和上下文预算，然后生成：

- `ground-truth-def.csv`：1500 个逻辑 batch，顺序为
  `D1,E1,F1,D2,E2,F2,...,D500,E500,F500`；当前 10 arms 会展开为 15000 个物理
  batch。
- `ground-truth-def-slice001-050.csv` 至
  `ground-truth-def-slice451-500.csv`：10 个可独立运行/恢复的切片；每片包含每种
  scenario 50 行，共 150 个逻辑 batch，对应 1500 个物理 batch。
- `single_scenario_d_500_report.html`、`single_scenario_e_500_report.html`、
  `single_scenario_f_500_report.html`：包含校验状态、token 统计、12 档分布图、上下文
  余量、源 CSV SHA-256 和 prompt 样例。
- `ground-truth-def-report.html`：包含三场景对比、逻辑/物理 batch 数、全部切片的
  行数、文件大小和 SHA-256。

所有 HTML 都是自包含静态文件，不依赖外部 JavaScript/CSS。总报告和 F 单场景报告
已在本地浏览器进行渲染检查，表格、统计卡片、长 hash 和 token 分布均正常显示。

#### 运行时代码同步

`ground_truth_experiment.py` 保留 A/B/C 并新增 D/E/F 的 request type、输入范围和
固定输出 token 校验。`validate_input_length()` 还会在真实推理前执行与数据准备脚本
一致的 2048-token 总上下文检查，防止手工修改 CSV 后在 decode 中途才出现越界。

`run.sh` 的默认输入切换为 `./dataset/ground-truth-def-slice001-050.csv`，默认输出
切换为 `outputs_ground_truth_def_slice001_050.csv`；`readme.md` 同步记录 D/E/F 的
输出长度、decode step 数和逐 slice 运行方式。同步三节点时需要同时复制更新后的
代码和计划运行的 DEF slice 到各节点项目的 `dataset/` 目录。

新增 `.gitattributes` 强制所有 shell 脚本保持 LF 行尾；`run.sh` 已规范化并通过
`bash -n`。这样从 Windows 工作区同步到 Linux 节点时，不会因 CRLF 破坏反斜杠续行。

本次只完成数据、报告和运行时代码准备，尚未启动新的三节点 ground-truth。建议先
运行 `ground-truth-def-slice001-050.csv`，确认 E/F 的 top-2 gap 与瓶颈 Rank 分布
符合预期后，再顺序运行剩余九个切片。

## 2026-08-17

### Ground Truth 全臂枚举实验

#### 实验目标

本次在 `test-groundtruth` 分支新增离线 ground truth 数据采集模式。它不执行
UCB、Contextual 或 Lipschitz 的在线选臂，也不更新任何 reward；对于输入数据集
的每一个逻辑 batch，依次使用 `scheduler.py` 中 `CANDIDATE_ARMS` 的全部实际臂
各推理一次，再根据同一个请求下的实测性能给所有臂排序。

臂数量没有写死为 20。设当前候选列表为

$$
\mathcal A=\texttt{CANDIDATE\_ARMS},\qquad K=|\mathcal A|,
$$

数据集共有 $D$ 个逻辑 batch，则总执行次数自动为

$$
N_{\mathrm{execution}}=D\times K.
$$

当前分支有 10 个候选臂，数据集有 150 行，因此会执行
$150\times10=1500$ 个物理 batch。以后只需增删 `CANDIDATE_ARMS`，循环次数、
日志行数和排名范围都会自动随 $K$ 变化，不需要同步修改其他常量。

#### 逻辑 Batch 与执行 Batch

`dataset_batch` 表示原始数据集的行号，同一条 prompt 在所有臂上保持相同；
`execution_batch` 是三节点通信、模型切层、`scheduler.csv` 和原始实验日志使用的
连续物理批次号。二者关系为

$$
\text{execution\_batch}=(\text{dataset\_batch}-1)K+
\text{execution\_order}.
$$

为减小固定先后顺序引入的温度、缓存或后台负载偏差，每个逻辑 batch 会循环
平移臂执行顺序，但 `arm_index` 始终表示该臂在 `CANDIDATE_ARMS` 中的固定位置。
同一个逻辑 batch 的全部臂使用同一个 Environment batch 状态，避免模拟带宽或
时延恰好在臂枚举中间变化。

#### 数据集制作

新增 `C:/Users/smbu/Desktop/lab7/dataset/prepare_ground_truth_dataset.py`。脚本分别
读取以下文件的前 50 条非空 prompt：

| Scenario | 源文件 | Input token 范围 | 固定输出 token |
|---|---|---:|---:|
| A / LISO | `single_scenario_a_500.csv` | 800-1500 | 90 |
| B / SILO | `single_scenario_b_500.csv` | 10-150 | 400 |
| C / MIMO | `single_scenario_c_500.csv` | 200-600 | 256 |

输出 `ground-truth-test.csv`，总计 150 行，顺序为
`A1,B1,C1,A2,B2,C2,...,A50,B50,C50`，字段为
`prompt,scenario,request_type,target_output_tokens,source_row`。

`ground_truth` 要求 `BATCH_SIZE=1`，并拒绝同时设置
`--force-decode-steps`。场景目标表示最终保存的 token ID 总数，prefill 得到的
first token 计入其中，因此

$$
\text{decode\_step\_count}=\text{target\_output\_tokens}-1.
$$

#### 实测评分与排名

每个 arm 完成后直接复用现有三节点 record 的模式相关时间。`distributed` 使用
`Tcompute_plus_Ttransfer_plus_Tcomm_ms`；`cloud-base` 使用
`Tdecode_plus_Ttransfer_plus_Tcomm_ms`。设三个 Rank 的时间为 $T_{0,t,a}$、
$T_{1,t,a}$、$T_{2,t,a}$，则瓶颈时间和每 token cost 为

$$
T^{\max}_{t,a}=\max(T_{0,t,a},T_{1,t,a},T_{2,t,a}),
$$

$$
C_{t,a}=\frac{T^{\max}_{t,a}}
{\max(1,\text{decode\_step\_count}_{t,a})}.
$$

仅为方便与现有 reward 数值范围对照，再计算不带探索项、不带历史平均的实测
score：

$$
S_{t,a}=\frac{1}{1+C_{t,a}/100}.
$$

同一个 `dataset_batch` 内按 $C_{t,a}$ 从小到大排序，cost 相同时按
`CANDIDATE_ARMS` 原始顺序打破平局；`arm_ranking=1` 即该请求的 ground truth
最优臂。

#### 日志与文件

- 新增 `ground_truth_experiment.py`：负责读取带标签数据、按实际 arm list 生成
  执行计划、校验场景输入长度、计算实测 score、完成组内排名以及写出结果。
- `bandit_logs/arm_details_ground_truth_YYYY-MM-DD-HH-MM.csv`：每个逻辑 batch
  完成全部 $K$ 个臂后立即追加 $K$ 行并执行 `flush + fsync`。字段包含逻辑/物理
  batch、场景、输入/输出长度、臂、三 Rank 时间、瓶颈时间、每 token cost、
  实测 score 和 ranking。
- `outputs_kv.csv`：新增逻辑 batch、执行 batch、场景和 arm 标识，避免重复
  prompt 的生成结果无法追溯到具体臂。
- `scheduler.py`：新增非学习型 `GroundTruthBanditPolicy`。它严格使用
  `CANDIDATE_ARMS`，不自动插入默认 `SPLIT_LAYERS`。即使默认 split 不在列表中，
  ground truth 仍直接从列表第一个臂开始，并且总臂数仍为 $K=|\mathcal A|$。
- `inference_loops.py`：把每个数据集行展开为 $K$ 次现有分布式推理；模型增量
  切层、NCCL 数据流、KV cache 和原始计时逻辑均复用，不另建通信协议。
- `config.py`、`run.sh`：新增并默认选择 `ground_truth`；`run.sh` 保留用户已改的
  `ground-truth-test.csv`、`BATCH_SIZE=1` 和 `MAX_INPUT_TOKENS=2000`。


## 2026-08-14

### 经典 UCB1 累计平均 Reward

#### 修改范围

本次只修改 `BANDIT_POLICY=ucb` 的 reward 更新与 score 计算。`contextual`、`contextual_controlled`、`lipschitz` 和 `contextual_lipschitz` 均保持原有行为，`UCB_EXPLORATION_WEIGHT=0.01` 及 `self.exploration_weight` 参数传递方式不变。

#### 修改前

普通 UCB 每次使用最新 batch 覆盖该 arm 的 reward：

$$
R_a\leftarrow r_t
$$

因此旧观测不会保留在 reward 中，score 的中心值只代表最后一次执行结果。

#### 单次观测

每个有效 batch 仍先计算三个 Rank 中瓶颈节点的每 decode step 耗时：

$$
C_t=
\frac{\max(T_{0,t},T_{1,t},T_{2,t})}
{\max(1,\mathrm{decode\_step\_count}_t)}
$$

再将 cost 映射为区间 $(0,1]$ 内、越大越好的单次 reward：

$$
r_t=\frac{1}{1+C_t/100}
$$

#### 累计平均 Reward

arm $a$ 完成第 $n$ 次有效 pull 后，使用经典样本均值更新：

$$
\bar R_{a,n}=
\frac{(n-1)\bar R_{a,n-1}+r_t}{n}
$$

代码中的 `stats["reward"]` 现在表示 $\bar R_{a,n}$。初始 `reward=0.5` 在 `pulls=0` 时不会进入第一次平均；第一次有效观测后有 $\bar R_{a,1}=r_1$。

`stats["mean_cost"]` 同步保存累计平均 cost，`stats["last_cost"]` 保存最新 cost，但 UCB1 选臂使用的是累计平均 reward。由于 reward 映射是非线性的，代码采用“逐批先计算 reward，再平均 reward”，而不是由平均 cost 反推 reward：

$$
\frac{1}{n}\sum_{i=1}^{n}\frac{1}{1+C_i/100}
\neq
\frac{1}{1+\bar C/100}
$$

#### UCB1 Score

所有 arm 至少获得一次有效观测后，score 使用经典 UCB1 形式：

$$
\operatorname{Score}_a=
\bar R_a+
c\sqrt{\frac{2\ln N}{N_a}}
$$

其中 $N$ 是所有 arm 的有效 pull 总数，$N_a$ 是 arm $a$ 的有效 pull 数，探索强度继续读取原有变量：

$$
c=\texttt{self.exploration\_weight}=0.01
$$

实际选臂与 `arm_details.score` 共用 `_ucb1_exploration_bonus()`，避免运行决策和日志审计采用不同公式。

#### 初始化与冷启动

- 若存在 `pulls=0` 的 arm，继续按 `CANDIDATE_ARMS` 顺序优先执行未观测 arm；全部 arm 至少完成一次有效 pull 后才比较 UCB1 score。
- 整次运行的第一个完成 batch 仍作为系统冷启动，不增加 `pulls`，不更新 reward，也不参与 score；第二个 batch 继续使用相同 arm，并形成第一个有效 UCB1 样本。
- 普通 UCB1 仍不包含 elimination，所有候选 arm 后续都可能因探索项而再次被选择。

#### 代码改动

- `scheduler.py`：将普通 UCB 的 `_update_latest_arm_cost()` 替换为 `_update_ucb_arm_reward()`，累计更新平均 reward。
- `scheduler.py`：新增 `_ucb1_exploration_bonus()`，统一实现 $0.01\sqrt{2\ln N/N_a}$。
- `scheduler.py`：实际选臂和 `arm_details.score` 均改为使用同一个 UCB1 探索项函数。
- `log.md`：记录累计平均 reward、UCB1 score、冷启动和未探索 arm 规则。

## 2026-08-11

### Contextual Controlled Policy

#### 实验目标

新增 `BANDIT_POLICY=contextual_controlled`，用于 `contextual_bandit_test_tinyllama_labeled.csv` 的两阶段实验。该文件固定为 900 行：前 600 行是 A/B/C 交替的受控学习数据，后 300 行是 100A、100B、100C 的无标签决策评估数据。

新 policy 继承现有 `ContextualBanditPolicy` 的 LinUCB 模型，但不改变原来的 `contextual` policy。学习阶段使用 label 只控制输出长度；评估阶段不允许 label 进入 Context 或选臂逻辑，并冻结前 600 批学到的模型。

#### 学习阶段：Batch 1-600

场景及固定输出 token ID 数：

| Scenario | Request type | Target output tokens |
|---|---|---:|
| A | `long_input_short_output` | 80 |
| B | `short_input_long_output` | 386 |
| C | `medium_input_medium_output` | 256 |

Rank 0 仍然先 tokenize 当前 prompt，并由实际输入 token 长度构造 Context：

$$
x_t=\left[1,\frac{L_t^{in}}{1500},\frac{\hat L_t^{out}}{512},\frac{B_t}{128}\right]
$$

其中 $\hat L_t^{out}$ 使用推断 request type 对应的 `80/386/256`。CSV label 不直接作为特征；代码会核对 prompt 长度推断的 request type 与 label 是否一致，避免错误标签污染实验。

学习阶段设置：

```text
CONTROLLED_LEARNING_BATCHES = 600
CONTROLLED_EXPECTED_ARM_COUNT = 20
CONTROLLED_CONTEXT_WARMUP_PULLS = 10
BATCH_SIZE = 1
```

在 A/B/C 逐行交替的数据顺序下，每个 arm 会得到：

$$
10A+10B+10C=30\text{ batches}
$$

20 个 arm 共计：

$$
20\times30=600\text{ learning batches}
$$

启动时会检查实际有效 arm 必须为 20 个且彼此不重复。若默认 split 不在 `CANDIDATE_ARMS`，Scheduler 自动插入后会得到 21 个 arm；若候选表存在重复组合，也会直接报错，防止训练覆盖和顺序悄悄偏移。

#### 精确输出控制

`generate_rows_for_prompts()` 和 `generate_rows_for_prompts_cloud_base()` 新增内部参数 `forced_output_tokens`。该参数只由 `contextual_controlled` 的 learning batch 设置，不修改旧 `--force-decode-steps` 的语义。

目标 $N$ 表示最终保存的 token ID 总数，prefill 产生的 first token 计入其中：

$$
\text{generated token IDs}=N,\qquad
\text{decode\_step\_count}=N-1
$$

受控阶段忽略 EOS并保持 batch 中所有行 active。生成结束后逐行检查 `len(generated_tokens)==N`，数量不符时立即报错。输出文本仍使用 `skip_special_tokens=True`，因此特殊 token 不一定显示在 `generated_text` 中，但内部 token ID 数量严格受控。

#### 评估阶段：Batch 601-900

阶段切换只根据固定 batch 边界完成。Batch 601 起不读取 CSV 的 `scenario`、`request_type` 或 `phase` 作为决策信息：

1. 仅从当前 prompt 的实际 token 长度推断 request type。
2. 构造 Context，并为所有 arm 计算 LinUCB score。
3. 选择当前 Context 下 score 最大的 arm。
4. 不设置 `forced_output_tokens`，恢复 EOS 或 `--max-new-tokens` 停止规则；默认上限仍为 512。
5. 冻结各 arm 的 $A_a,b_a$、pulls 和 reward，不再用评估数据更新模型。

第一次进入 evaluation 时，会确认 20 个 arm 在三种 request type 下都至少完成 10 次学习观测；未完成则拒绝继续评估。

#### 防止 Label 泄漏

- Learning：label 只决定固定输出长度；选臂特征来自真实 prompt token 长度。
- Evaluation：Context 中 `scenario` 为空、`label_used=0`、`target_output_tokens=None`；CSV label 字段不参与类型推断、score 或选臂。
- `ContextualControlledBanditPolicy.select_arm()` 在 evaluation 中直接计算 LinUCB score，绕开 contextual warmup 的未完成 arm 搜索。
- `ContextualControlledBanditPolicy.update_after_batch()` 在 evaluation 中不修改任何线性模型参数。

#### arm_details 审计字段

`arm_details` 新增以下通用列；非 `contextual_controlled` policy 保持为空：

| 字段 | 含义 |
|---|---|
| `phase` | `learning` 或 `evaluation` |
| `label_used` | learning 为 1，evaluation 为 0 |
| `scenario` | learning 为 A/B/C，evaluation 为空 |
| `inferred_request_type` | 根据 prompt token 长度推断的请求类型 |
| `target_output_tokens` | learning 为 80/386/256，evaluation 为 `natural(max=512)` |
| `model_updated` | learning 为 1，evaluation 为 0 |

#### 文件改动

- `csv_io.py`：新增 labeled CSV 读取和必需列检查。
- `scheduler.py`：新增受控场景配置、Context 构造与 `ContextualControlledBanditPolicy`。
- `inference_loops.py`：接入 per-batch 固定输出、900行/Batch Size/参数检查，以及 evaluation 自然生成。
- `config.py`：新增 `contextual_controlled` policy 选项。
- `readme.md`：新增运行配置、数据格式和两阶段行为说明。
- `log.md`：记录本次设计、公式、阶段隔离和审计字段。

## 2026-08-10

### Environment 取消通信限制并保留变化点

- 保留 `DEFAULT_SCHEDULE` 中 batch 10 和 batch 50 两个环境变化点，方便后续网络变化实验直接在原位置填写参数。
- batch 10 和 batch 50 的五条有向链路均改为 `Bandwidth=None`，表示不施加应用层带宽上限。
- 两个变化点的 `time_comm_delay` 均改为 `0.0 ms`，表示不增加模拟单向通信延迟。
- `Environment.apply_batch()`、Rank 0 广播、Scheduler 环境快照和通信计时逻辑保持不变。本次只取消当前实验参数限制，没有删除动态变化接口。

### UCB算法重设计

本次直接重写现有 `LayerBanditPolicy` 的普通 UCB 行为，没有新增 UCB policy 类。Contextual Bandit 保持自己的逐批线性模型更新；Lipschitz Bandit 继续使用原有窗口统计入口，因此本次规则只对 `BANDIT_POLICY=ucb` 生效。

#### 改动前

- 每个 arm 使用 `window_size=2`、`warmup_skip=1`：连续运行两个 batch，跳过第一个 batch，只用第二个 batch 形成一次有效 pull。
- 每次获得新 cost 后，与该 arm 的历史 cost 计算累计平均值：

  $$
  \bar C_{a,n}=\frac{(n-1)\bar C_{a,n-1}+C_{a,n}}{n}
  $$

- reward 由历史平均 cost 生成，因此一次新的环境或性能变化会被旧数据平滑：

  $$
  R_{a,n}=\frac{1}{1+\bar C_{a,n}/100}
  $$

#### 改动后

- 整次运行只有第一个完成的 batch 是全局预热。该 batch 不增加 `pulls`，不更新 cost/reward，也不使用 score 选下一个 arm；下一批继续使用相同的默认 arm。
- 从第二个 batch 开始，每完成一个 batch 就形成一次有效 pull，并立即选择下一批使用的 arm。切换到新 arm 后不再额外跳过预热 batch。
- cost 仍使用三个 Rank 中最慢节点的每 decode step 耗时：

  $$
  C_t=\frac{\max(T_0,T_1,T_2)}{\max(1,\text{decode\_step\_count})}
  $$

- 取消普通 UCB 的历史平均 reward。每次运行直接用最新 batch 覆盖该 arm 的 cost 和 reward：

  $$
  R_a\leftarrow R_t=\frac{1}{1+C_t/100}
  $$

- `pulls` 仍然累计，用于 UCB 探索项；score 使用该 arm 的最新 reward：

  $$
  \operatorname{Score}(a)=R_{a,\mathrm{latest}}+0.01\sqrt{\frac{\ln N}{N_a}}
  $$

- `exploration_weight` 默认值明确为 `0.01`。未观测 arm 优先执行的原有初始化机制保持不变。
- `arm_details` 在全局预热 batch 中保留 arm 行，但 `reward` 和 `score` 为空；第二个 batch 起正常记录最新 reward 和 UCB score。

#### 与旧 UCB 的核心区别

| 项目 | 改动前 | 改动后 |
|---|---|---|
| 预热范围 | 每个 arm 的统计窗口都跳过第一个 batch | 整次运行只跳过 Batch 1 |
| 更新频率 | 每两个 batch 更新一次 | Batch 2 起每个 batch 更新一次 |
| reward 数据 | arm 历史平均 cost | arm 最近一次 cost |
| 环境变化响应 | 较慢，旧观测持续影响 | 较快，最新观测立即覆盖 |
| 探索强度 | `0.01` | `0.01`，明确为普通 UCB 默认值 |

### 加权 Lipschitz Bandit 与在线距离学习

#### 设计目标

本小节只记录 `BANDIT_POLICY=lipschitz` 的独立设计；普通 UCB 的当前行为以同日上方“UCB算法重设计”为准。Contextual Bandit 和预留的 Contextual + Lipschitz 接口不受本小节影响。Lipschitz 版本解决三个问题：两个切分点对异构节点的影响不应被视为相同；未执行过的 arm 不再被强制逐一探索；Lipschitz policy 不读取 A/B/C 请求类型或其他 context。

#### 1. 加权 arm 距离

对三节点切分 arm

$$
a=(p_1,p_2)
$$

先计算归一化坐标差：

$$
x_1(a,b)=\frac{|p_1^a-p_1^b|}{N_{\mathrm{layers}}},\qquad
x_2(a,b)=\frac{|p_2^a-p_2^b|}{N_{\mathrm{layers}}}
$$

不再固定使用相同权重，而是学习两个有效斜率：

$$
D_q(a,b)=q_1x_1(a,b)+q_2x_2(a,b)
$$

其中，$q_1$ 表示第一个切分点变化对 reward 的敏感程度，$q_2$ 表示第二个切分点变化对 reward 的敏感程度。初始化为：

$$
q_1=q_2=0.5
$$

这与旧实现中的 $0.5(x_1+x_2)$ 惩罚保持一致。为了便于解释和审计，同时派生：

$$
L=q_1+q_2,\qquad
w_1=\frac{q_1}{q_1+q_2},\qquad
w_2=\frac{q_2}{q_1+q_2}
$$

因此：

$$
D_q(a,b)=L\left(w_1x_1(a,b)+w_2x_2(a,b)\right)
$$

代码直接学习 $q_1,q_2$，避免同时学习 $L,w_1,w_2$ 时出现同一距离可以由多组缩放参数表示的问题。

#### 2. q1/q2 在线更新

每完成一次有效 pull，只使用已经有 reward 的 arm 与本次更新 arm 组成观测对。对观测 arm $a_i,a_j$：

$$
y=\gamma\left|\bar R(a_i)-\bar R(a_j)\right|,\qquad
\hat y=q_1x_1(a_i,a_j)+q_2x_2(a_i,a_j)
$$

其中安全系数默认 $\gamma=1.1$。当距离惩罚低估 reward 差异时快速增大斜率；高估时缓慢减小，防止置信区间因一次噪声观测突然收缩：

$$
q_k\leftarrow\operatorname{clip}\left(
q_k+\eta_+[y-\hat y]_+x_k-\eta_-[\hat y-y]_+x_k,
q_{\min},q_{\max}
\right)
$$

当前默认值：

```text
eta_up = 0.05
eta_down = 0.005
q_min = 0.001
q_max = 10.0
gamma = 1.1
```

#### 3. 取消未探索 arm 强制优先

旧版本遇到 `pulls == 0` 的 arm 会直接返回该 arm，因此必须完整遍历所有候选臂。新版本删除这条规则，未执行过的 arm 也通过已观测 arm 推导出的 Lipschitz 置信区间参与选择。

对已观测 arm $i$，统计置信半径：

$$
\beta_i=c\sqrt{\frac{\ln(\max(N,2))}{N_i}}
$$

其中 $c$ 为 `exploration_weight`，$N$ 为有效 pull 总数，$N_i$ 为 arm $i$ 的 pull 数。候选 arm $a$ 的置信下界和上界为：

$$
\operatorname{LCB}(a)=
\max_{i\in\mathcal O}\left[\bar R_i-\beta_i-D_q(a,i)\right]
$$

$$
\operatorname{UCB}(a)=
\min_{i\in\mathcal O}\left[\bar R_i+\beta_i+D_q(a,i)\right]
$$

其中 $\mathcal O$ 是至少完成过一次有效 pull 的 arm 集合。尚无任何观测时，置信区间使用 $[0,1]$；第一份有效观测仍由当前默认切分产生，但不会触发“依次尝试全部未探索 arm”的循环。

每轮选择当前 active arm 中上界最大的 arm：

$$
a_{t+1}=\arg\max_{a\in\mathcal A_{\mathrm{active}}}\operatorname{UCB}(a)
$$

#### 4. active_arms

每次有效 pull 后，根据最新 reward、$q_1/q_2$ 和置信区间重新计算 active 集合：

$$
\mathcal A_{\mathrm{active}}=
\left\{a:\operatorname{UCB}(a)+\epsilon\ge
\max_b\operatorname{LCB}(b)\right\}
$$

当前 $\epsilon=0$。该集合不是永久删除列表，而是每轮从全部候选 arm 重算；随着新观测、斜率或置信区间变化，之前 inactive 的 arm 可以重新激活。

#### 5. 非 Contextual 约束

`LipschitzBanditPolicy` 继续继承非 contextual 的基础 policy，不读取 `request_type`、输入 token 数、预计输出 token 数、batch size 或 context vector。它只使用统一的每 token 成本 reward、arm 坐标和历史 pull 统计。混合 A/B/C 请求会共同更新同一个 arm 模型，这是本策略刻意保留的行为。

#### 6. arm_details 审计增强

`bandit_logs/arm_details_YYYY-MM-DD-HH-MM.csv` 仍然在每个 batch 后为所有候选 arm 写一行，但新增 `pull_completed` 用于识别该 batch 是否真正完成了一个统计窗口。热身 batch 也保留快照，变化值为 0；当 `pull_completed=1` 时，可以完整审计一次 policy 更新。

主要新增字段如下：

| 字段 | 含义 |
|---|---|
| `policy` | 当前 policy 名称；本设计对应 `lipschitz` |
| `pull_completed` | 当前 batch 是否完成一次有效 pull 更新 |
| `pull_index` | 当前累计有效 pull 序号 |
| `updated_arm` | 本次直接接收新成本/reward 的 arm |
| `pulls_before`, `pulls`, `pulls_delta` | 每个 arm 更新前、更新后和变化的 pull 数 |
| `reward_before`, `reward`, `reward_delta` | 每个 arm 的直接 reward 更新变化；通常只有 `updated_arm` 直接变化 |
| `score_before`, `score`, `score_delta` | 每个 arm 的 Lipschitz 置信上界变化；即使 direct reward 不变，也可能因 $q_1/q_2$ 或其他 arm 的观测而变化 |
| `confidence_lower`, `confidence_upper`, `confidence_width` | 每个 arm 当前置信区间及宽度 |
| `q1`, `q2`, `q1_delta`, `q2_delta` | 当前有效斜率及本次 pull 的变化量 |
| `lipschitz_constant`, `w1`, `w2` | 由 $q_1/q_2$ 派生的 $L,w_1,w_2$ |
| `active` | 当前行 arm 是否属于 active 集合 |
| `active_arms` | 当前全部 active arm 的完整列表 |
| `selected` | 当前完成 batch 实际使用的 arm |
| `next_selected` | policy 为下一统计窗口选中的 arm |

普通 UCB 和 Contextual policy 仍使用原有 reward/score 逻辑；它们不具备含义的 Lipschitz 专用审计列保持为空。

#### 修改文件

- `scheduler.py`：重写 `LipschitzBanditPolicy` 的距离、在线斜率学习、置信区间、active arm 和选臂逻辑；扩展 `arm_details` 输出结构。
- `log.md`：记录本次算法设计、公式、默认参数和审计字段。

## 2026-08-06

### TinyLlama 动态切臂改为增量加载

- 参考 `C:\Users\smbu\Desktop\lab7\dist` 中 qwen2-7b 版本的实现，为 tinyllama 版本补上增量 decoder layer 切换能力。动态切臂时不再释放旧模型分区后重新 lazy load 整个 rank stage，而是保留当前 rank 已经加载过的层。
- 新增 `incremental_layer_partition.py`，由 `IncrementalLayerPartition` 管理本 rank 的 active / inactive decoder layers：
  - 新 split 与旧 split 重叠的层直接保留在 `model.model.layers` 中。
  - 旧 split 移出的层从 active `ModuleList` 移到本地缓存字典，后续切回时可直接命中。
  - 新 split 需要但本地从未加载过的层，才从 safetensors checkpoint 读取。
  - CUDA 显存不足时，会按 least-recently-used 方式清理 inactive layer，并保留目标 split 需要的层。
- `rank0_generate_dynamic()` 和 `pipeline_serve_dynamic()` 已接入增量切换：
  - 首个 batch 仍通过 `load_model_part(..., lazy_load=True)` 加载初始分区。
  - 后续 batch 如果 split 不变，继续复用当前模型分区。
  - 后续 batch 如果 split 变化，调用 `layer_partition.switch_to(layer_start, layer_end, batch_number)`，并打印 `retained`、`cache_hits`、`local_loaded`、`inactive_cache`、`evicted` 和 `elapsed_ms`，便于观察切臂是否真的只补加载缺失层。
- 恢复动态加载后的 model-ready barrier。每个 batch 在所有 rank 完成本地加载或增量切换后，才进入 hidden states / KV cache 等 NCCL 通信，避免某个 rank 还在读 checkpoint 时其他 rank 已经进入 P2P recv/send。
- cloud-base 模式下，Rank 2 的完整 prefill model 也会在 barrier 前完成首次加载，避免 Rank 0 / Rank 1 已经开始等待 KV cache，而 Rank 2 仍在加载完整模型造成计时或阻塞混淆。
- `model_loader.py` 补齐 qwen2-7b 版本中的加载器改动：
  - 新增 `resolve_model_dtype_from_config()`，当 `--dtype auto` 时先从 config 中解析真实 dtype，再用于模型加载和通信 dtype。
  - 新增 `validate_model_structure()`，提前确认模型具有 Llama-style pipeline 所需的 `model.layers`、`embed_tokens`、`norm` 和 `lm_head`。
  - lazy loader 改为只构造当前 rank 需要的 decoder layer 数量，再用 `original_checkpoint_key()` 将本地 layer index 映射回 checkpoint 的全局 layer id，减少初始化和切换开销。

### Files changed

- `incremental_layer_partition.py`: 新增本地层缓存、缺失层按需加载、inactive layer LRU 清理和切臂结果统计。
- `inference_loops.py`: 动态推理路径从“split 变化时释放并重载”改为“split 变化时增量切换”，并加入 model-ready barrier。
- `model_loader.py`: 补上 dtype 解析、结构校验，以及只构造 rank-local stage 的 lazy load 路径。
- `distributed_tinyllama_inference.py`: 动态/静态入口统一使用解析后的模型 dtype 和通信 dtype。
- `scheduler.py`: 补上 `CONTEXT_WARMUP_PULLS`，contextual bandit 预热阶段改为“每个请求类型下每个实际可用 arm 至少测试 N 次”。候选 arm 数量仍由 `CANDIDATE_ARMS` 和 `_valid_arm()` 根据实际 `total_layers` 过滤决定，不把 warmup 轮数和固定 arm 数量绑死。

### Verification

- 已使用 Codex bundled Python 对以下文件做语法编译检查：

  ```text
  distributed_tinyllama_inference.py
  inference_loops.py
  model_loader.py
  incremental_layer_partition.py
  scheduler.py
  ```

- 未在本机执行 NCCL 多 rank 推理验证；该验证需要实际三节点 / GPU 运行环境。

## 2026-07-31

### Context-aware warmup for Contextual Bandit

- Fixed the contextual warmup logic so that exploration is grouped by request context instead of only by global arm pulls.
- Previous contextual warmup used only:

  ```text
  stats[arm]["pulls"] == 0
  ```

  This means each arm was tried once globally. With an A/B/C mixed dataset, the first few batches could look like:

  ```text
  A -> arm1
  B -> arm2
  C -> arm3
  A -> arm4
  ...
  ```

  After every arm had one global observation, the policy immediately switched to score-based selection. That did not guarantee that every arm had been observed under every request type.
- New contextual warmup tracks observations with:

  ```text
  context_arm_pulls[request_type][arm]
  ```

  where `request_type` is inferred from the current batch context:

  ```text
  long_input_short_output
  short_input_long_output
  medium_input_medium_output
  ```

- The new selection rule for `BANDIT_POLICY=contextual` is:

  ```text
  request_type = context.request_type

  if any arm has context_arm_pulls[request_type][arm] == 0:
      choose the first untried arm for this request_type
  else:
      choose argmax LinUCB score under the current context
  ```

- This makes the intended A/B/C learning stage possible. For example, with six arms and `batch_size=1`, the first eighteen batches can cover:

  ```text
  A -> arm1, B -> arm1, C -> arm1
  A -> arm2, B -> arm2, C -> arm2
  ...
  A -> arm6, B -> arm6, C -> arm6
  ```

- Plain `BANDIT_POLICY=ucb` is unchanged. It still uses the global arm-pull logic and does not read request context.
- `ContextualLipschitzBanditPolicy` inherits this contextual warmup behavior because it currently extends `ContextualBanditPolicy`.

### Files changed

- `scheduler.py`: added `context_arm_pulls`, context-key extraction, and per-context warmup selection inside `ContextualBanditPolicy`.
- `log.md`: documented the 2026-07-31 contextual warmup correction separately from the earlier 2026-07-30 bandit changes.

## 2026-07-30

### Bandit reward, candidate arms, and Lipschitz policy

#### 1. Contextual warmup exploration

- Fixed the first contextual implementation's tendency to stay at the default split `(5, 15)`.
- Cause: all arms start with the same model state:

  $$
  A_a = \lambda I,\quad b_a = 0,\quad \theta_a = 0
  $$

  Therefore all arms initially have the same LinUCB score under the same context. The previous tie-break selected the first arm in the sorted candidate list, which is the default split `(5, 15)`. After `(5, 15)` received the first positive reward, it could keep winning against untried arms.
- New rule:

  ```text
  if any arm has pulls = 0:
      choose the first untried arm
  else:
      choose argmax LinUCB score
  ```

- The default exploration parameters were also reduced:

  ```text
  window_size = 2
  exploration_weight = 0.01
  ```

  `window_size=2` means a selected arm is updated after two consecutive completed batches in the plain UCB path. In contextual mode, each completed selected arm updates its linear model directly, while `exploration_weight=0.01` controls the LinUCB uncertainty bonus after all arms have at least one observation.

#### 2. Unified reward function

- Removed the old relative normalized reward from plain `ucb`.
- Previous `ucb` reward depended on the currently observed arm set:

  $$
  reward(a)
  =
  1
  -
  \frac{\bar{C}_a - C_{\min}}{C_{\max} - C_{\min}}
  $$

  This has been removed because the reward scale changes when new arms are observed or when the request context changes.
- All policies now use the same cost and reward definition. First, compute bottleneck cost per decode step:

  $$
  T_t = \max_r T_{t,r}
  $$

  $$
  D_t = \max(1, decode\_step\_count_t)
  $$

  $$
  C_t = \frac{T_t}{D_t}
  $$

- Then convert it to an absolute bounded reward:

  $$
  r_t =
  \frac{1}{1 + \frac{C_t}{\tau}}
  $$

  with:

  $$
  \tau = 100 ms
  $$

- For plain `ucb`, the window cost is now the mean of per-token bottleneck costs inside the reward window, and the stored reward is computed from that absolute cost. For `contextual`, each completed selected arm already uses the same per-token cost and absolute reward.

#### 3. Explicit candidate arm set for early algorithm tests

- Replaced offset-based arm generation with an explicit candidate arm list in `scheduler.py`.
- The current candidate set is:

  ```text
  CANDIDATE_ARMS = [
      (1, 11),
      (1, 15),
      (1, 19),
      (5, 11),
      (5, 15),
      (5, 19),
      (9, 11),
      (9, 15),
      (9, 19),
  ]
  ```

- This keeps the search space fixed across experiments. Changing `SPLIT_LAYERS` no longer silently changes the candidate arm set. Each arm is still validated by:

  $$
  0 < p_1 < p_2 < total\_layers
  $$

  Invalid arms are skipped automatically for models with fewer layers.

#### 4. Simple Lipschitz-UCB implementation

- Implemented a first runnable `LipschitzBanditPolicy`. It keeps the same before-batch selection and after-batch update timing as the other policies.
- Reward update is unchanged and reuses the unified per-token absolute reward:

  $$
  C_t = \frac{\max_r T_{t,r}}{\max(1, decode\_step\_count_t)}
  $$

  $$
  r_t = \frac{1}{1 + \frac{C_t}{\tau}}
  $$

- The Lipschitz policy changes only the arm score. Each arm is:

  $$
  a = (p_1, p_2)
  $$

  and the normalized distance between two arms is:

  $$
  d(a,b)
  =
  \frac{|p_1^a - p_1^b| + |p_2^a - p_2^b|}{total\_layers}
  $$

- For a candidate arm `a`, the Lipschitz-smoothed reward estimate is:

  $$
  \hat{r}_{lip}(a)
  =
  \max_{b \in \mathcal{O}}
  \left(
  r(b) - L \cdot d(a,b)
  \right)
  $$

  where:

  $$
  \mathcal{O} = \{b \mid pulls(b) > 0\}
  $$

  is the set of observed arms, and `L` is the Lipschitz constant.
- Current implementation uses:

  $$
  L = 0.5
  $$

- After all arms have been tried at least once, the selected arm maximizes:

  $$
  score(a)
  =
  \hat{r}_{lip}(a)
  +
  \alpha
  \sqrt{\frac{\log N}{n_a}}
  $$

  where:

  $$
  N = total\_pulls,\quad n_a = pulls(a)
  $$

- Before all arms have observations, the policy keeps the same warmup behavior:

  ```text
  if any arm has pulls = 0:
      choose the first untried arm
  ```

- `arm_details.score` for `BANDIT_POLICY=lipschitz` now records the Lipschitz-UCB score above. `arm_details.reward` remains the direct observed reward of that arm, not the smoothed estimate.

### Files changed

- `scheduler.py`: added contextual warmup exploration, replaced relative UCB reward with unified per-token absolute reward, changed candidate arms to an explicit list, and implemented the first runnable `LipschitzBanditPolicy`.
- `log.md`: separated the 2026-07-30 bandit changes from the earlier 2026-07-29 contextual-bandit design notes.

## 2026-07-29

### current-batch arm selection and Contextual Bandit

#### 1. Arm selection timing change

- Changed the scheduler control loop from "previous batch selects next batch" to "current batch context selects current batch".
- Previous timing:

  ```text
  run batch N
  -> collect records for batch N
  -> update Bandit
  -> select arm for batch N+1
  -> reallocate_layer(batch=N+1)
  ```

- New timing:

  ```text
  Rank 0 reads prompts
  -> precompute context for every prompt batch

  before running batch N:
      context_N is loaded
      policy selects arm_N from context_N and historical observations
      reallocate_layer(batch=N, arm=arm_N)
      Rank 0 broadcasts boundaries_N

  after running batch N:
      Rank 0 collects all rank timing records
      scheduler stores summary_N
      policy updates from context_N, arm_N, reward_N
      arm_details logs the post-update policy snapshot
  ```

- This timing now applies to every policy. Plain `ucb` ignores the context vector but still selects before the current batch. `contextual` uses the current batch context directly. `reallocate_layer()` remains only the writer: it converts the selected arm into layer boundaries and writes the current batch row in `scheduler.csv`.

#### 2. Contextual Bandit design

- Added an active `ContextualBanditPolicy` implementation in `scheduler.py`. The policy is enabled by setting:

  ```bash
  BANDIT_POLICY=${BANDIT_POLICY:-contextual}
  ```

  `run.sh` passes this as `--bandit-policy` on Rank 0 only. The default remains:

  ```bash
  BANDIT_POLICY=${BANDIT_POLICY:-ucb}
  ```

- Rank 0 builds one context vector for each prompt batch after reading the input CSV. No extra dataset file and no extra worker metric tensor field is required.
- The request type is inferred from input token length:

  ```text
  short_input_long_output:    L_in in [10, 150],    L_out_est = 384
  medium_input_medium_output: L_in in [200, 600],   L_out_est = 256
  long_input_short_output:    L_in in [800, 1500],  L_out_est = 72
  ```

  If a batch falls outside these intervals, it is assigned to the nearest interval. `L_out_est` is therefore inferred from `L_in`; it is not passed as a separate input.
- The contextual feature vector is:

  $$
  x_t =
  \begin{bmatrix}
  1 \\
  L_{in,norm} \\
  L_{out,norm} \\
  B_{norm}
  \end{bmatrix}
  $$

  with:

  $$
  L_{in,norm} = \min\left(\frac{L_t^{in}}{1500}, 1\right)
  $$

  $$
  L_{out,norm} = \min\left(\frac{\hat{L}_t^{out}}{512}, 1\right)
  $$

  $$
  B_{norm} = \min\left(\frac{B_t}{128}, 1\right)
  $$

- Each arm is still a three-rank layer split:

  $$
  a = (p_1, p_2)
  $$

  $$
  rank_0 = [0, p_1), \quad rank_1 = [p_1, p_2), \quad rank_2 = [p_2, L)
  $$

- Each arm maintains an independent linear model:

  $$
  A_a = \lambda I
  $$

  $$
  b_a = 0
  $$

  $$
  \theta_a = A_a^{-1} b_a
  $$

- Before batch `t`, Contextual Bandit selects:

  $$
  score_t(a)
  =
  \theta_a^T x_t
  +
  \alpha
  \sqrt{x_t^T A_a^{-1} x_t}
  $$

  $$
  a_t = \arg\max_a score_t(a)
  $$

  where `alpha` currently reuses the existing `exploration_weight` value.
- Contextual Bandit does have an exploration term. In the score above:

  $$
  \alpha \sqrt{x_t^T A_a^{-1} x_t}
  $$

  is the uncertainty bonus. The first part,

  $$
  \theta_a^T x_t
  $$

  is exploitation, meaning the predicted reward under the current linear model. The second part is exploration, meaning arms with less certainty under the current context get a higher score. Larger `exploration_weight` / `alpha` explores more aggressively; smaller values make the policy greedier.
- After batch `t`, Rank 0 computes the observed bottleneck cost:

  $$
  T_t = \max_r T_{t,r}
  $$

  $$
  D_t = \max(1, decode\_step\_count_t)
  $$

  $$
  C_t = \frac{T_t}{D_t}
  $$

  Here `T_{t,r}` is the same mode-dependent rank time stored in `scheduler_summary`: for `distributed`, it is `T_comp + T_transfer + T_comm`; for `cloud-base`, it is `T_decode + T_transfer + T_comm`.
- The bounded reward is:

  $$
  r_t =
  \frac{1}{1 + \frac{C_t}{\tau}}
  $$

  with `tau = 100 ms` in this first implementation. Larger per-token bottleneck cost gives a smaller reward.
- Only the selected arm's model is updated:

  $$
  A_{a_t} \leftarrow A_{a_t} + x_t x_t^T
  $$

  $$
  b_{a_t} \leftarrow b_{a_t} + r_t x_t
  $$

- `ContextualLipschitzBanditPolicy` was reserved as the future combined policy hook. The runnable Lipschitz implementation was added later in the 2026-07-30 section.

### Files changed

- `scheduler.py`: added context construction helpers, changed policy timing to `select_arm()` before batch and `update_after_batch()` after batch, implemented `ContextualBanditPolicy` with a per-arm linear model, and kept compatibility wrappers for old method names.
- `inference_loops.py`: Rank 0 now precomputes batch contexts after reading prompts, selects the current batch arm before broadcasting boundaries, and updates the policy after records are collected.
- `config.py`: added `--bandit-policy`.
- `run.sh`: added `BANDIT_POLICY` and passes it on Rank 0.
- `log.md`: documented the new selection timing and Contextual Bandit formula.

### bandit policy extension hooks retained

- The policy registry remains the single place for switching scheduler algorithms:

  ```text
  ucb -> LayerBanditPolicy
  contextual -> ContextualBanditPolicy
  lipschitz -> LipschitzBanditPolicy
  contextual_lipschitz -> ContextualLipschitzBanditPolicy
  ```

- At the 2026-07-29 checkpoint, `lipschitz` was only a UCB-compatible placeholder with `arm_distance()`.
- At the 2026-07-29 checkpoint, `contextual_lipschitz` inherited the contextual LinUCB implementation and kept `arm_distance()` for later Lipschitz smoothing.

## 2026-07-15

### cold start analyzer

- Added `cold_start_analyzer.py` as an offline log analyzer. It does not change the distributed inference path.
- Input:

  ```text
  logs/log_YYYY-MM-DD-HH-MM.txt
  ```

- Output:

  ```text
  logs/cold_start_time_log_YYYY-MM-DD-HH-MM.csv
  ```

- The analyzer reads `prefill_mode` and `layer_allocation` from `--- summary after batch ...`, then joins them with each rank's `--- record ---` block by `batch` and `rank`.
- A cold-start estimate is emitted only when a full layer allocation appears for the first time and the immediately following batch uses the exact same allocation:

  ```text
  batch N:   layer_allocation=A
  batch N+1: layer_allocation=A
  ```

- Non-consecutive repeated allocations are ignored. This prevents comparing samples that have gone through another model partition in between.
- Each emitted row is still rank-local and includes:

  ```text
  prefill_mode, layer_allocation, rank, layer_start, layer_end,
  batch_size, input_seq_len_max, first_batch, second_batch,
  cold_start_type, first_observed_ms, second_observed_ms, cold_start_ms
  ```

- For `PREFILL_MODE=distributed`, cold start is estimated from prefill compute time:

  ```text
  cold_start_ms = max(0, first.prefill_comp_time_ms - second.prefill_comp_time_ms)
  ```

- For `PREFILL_MODE=cloud-base`, cold start is estimated from decode compute time after normalizing for different decode step counts:

  ```text
  stable_decode_ms_per_step = second.decode_comp_time_ms / second.decode_step_count
  expected_first_decode_ms = stable_decode_ms_per_step * first.decode_step_count
  cold_start_ms = max(0, first.decode_comp_time_ms - expected_first_decode_ms)
  ```

- Usage:

  ```bash
  python cold_start_analyzer.py logs/log_YYYY-MM-DD-HH-MM.txt
  ```

## 2026-07-09

### multi-arm bandit algorithm

- Added the first runnable online multi-arm bandit scheduler inside `scheduler.py`. The goal of this version is to close the adaptive scheduling loop with a small, auditable algorithm before adding more complex policies.
- Added a dedicated Bandit log directory:

  ```text
  bandit_logs/
  ```

  This directory stores Scheduler and Bandit observation files for the current run. These files are for inspection and later analysis only. The online Bandit decision still uses in-memory data and does not read these CSV files during inference.

- Moved the Scheduler summary file from the project root into `bandit_logs/`, and added a run timestamp to the filename:

  ```text
  bandit_logs/scheduler_summary_YYYY-MM-DD-HH-MM.csv
  ```

  The file content is unchanged:

  ```text
  batch,prefill_mode,layer_allocation,rank,time_label,time_ms
  ```

- Added a compact per-batch arm score file:

  ```text
  bandit_logs/arm_details_YYYY-MM-DD-HH-MM.csv
  ```

  After every completed batch, this file appends one row for every candidate arm. The current fields are:

  ```text
  batch,arm,reward,score,selected
  ```

  Field meaning:

  ```text
  batch
      The completed batch that triggered this snapshot.

  arm
      Candidate layer split in the form (p1,p2).

  reward
      Current normalized reward of the arm.

  score
      Current UCB score of the arm. Unmeasured arms are written as "untried"
      because the policy gives them priority before numeric UCB comparison.

  selected
      1 if this arm was selected for the next batch, otherwise 0.
  ```

- The code now separates the three scheduler responsibilities:

  ```text
  collect_batch_summary()
      -> collect the completed batch data into Scheduler memory

  run_bandit_after_batch()
      -> run the bandit policy and return the next arm

  reallocate_layer()
      -> write the selected next-batch allocation to scheduler.csv
  ```

- `reallocate_layer()` is intentionally not the decision maker. It is now the final writer: it receives an arm selected by the bandit, converts that arm into layer boundaries, validates the boundaries, and writes the next `scheduler.csv` row.
- Added `LayerBanditPolicy` to `scheduler.py`. This class owns the bandit state in memory:

  ```text
  current_arm
  active_batches
  per-arm pulls
  per-arm mean_cost
  per-arm normalized reward
  total_pulls
  ```

- The algorithm currently targets the three-rank layout. One arm is:

  $$
  arm_t = (p_1, p_2)
  $$

  It maps to:

  $$
  \begin{aligned}
  \text{rank0} &: [0, p_1) \\
  \text{rank1} &: [p_1, p_2) \\
  \text{rank2} &: [p_2, L)
  \end{aligned}
  $$

  where \(L\) is the total number of decoder layers.

- For TinyLlama with `total_layers=22`, an example arm is:

  $$
  arm_t = (5, 15)
  $$

  $$
  \begin{aligned}
  \text{rank0} &: [0, 5) \\
  \text{rank1} &: [5, 15) \\
  \text{rank2} &: [15, 22)
  \end{aligned}
  $$

- Candidate arms are generated as a small local search space around the default `--split-layers` value. For example, if the default split is `(5, 15)`, the policy tries nearby values using offsets `-4, -2, 0, +2, +4`, while keeping only valid arms satisfying:

  $$
  0 < p_1 < p_2 < L
  $$

- One arm is measured for six completed batches:

  $$
  W = 6
  $$

  The first batch is treated as warmup, and only the following five batches update the reward:

  $$
  \mathcal{B}_{reward} = \{b_2, b_3, b_4, b_5, b_6\}
  $$

  This is meant to reduce the effect of model reload, CUDA warmup, lazy-load behavior, and first-communication setup noise.

- For each completed batch `b`, the scheduler already has one summary time per rank in memory:

  $$
  T_{b,0},\quad T_{b,1},\quad T_{b,2}
  $$

  In `PREFILL_MODE=distributed`, this time is:

  $$
  T_{b,r}
  =
  \texttt{Tcompute\_plus\_Ttransfer\_plus\_Tcomm\_ms}_{b,r}
  $$

  This is the active value printed in the `Distributed` summary section for rank \(r\) in batch \(b\).

  In `PREFILL_MODE=cloud-base`, this time is:

  $$
  T_{b,r}
  =
  \texttt{Tdecode\_plus\_Ttransfer\_plus\_Tcomm\_ms}_{b,r}
  $$

  This is the active value printed in the `Cloud-base` summary section for rank \(r\) in batch \(b\).

- The cost of one batch is the slowest rank in that batch:

  $$
  C_b = \max(T_{b,0}, T_{b,1}, T_{b,2})
  $$

  This matches pipeline behavior: the slowest stage is the batch bottleneck.

- The cost of one arm is the mean bottleneck cost over the five non-warmup batches:

  $$
  cost(arm_t)
  =
  \frac{1}{5}
  \sum_{b \in \mathcal{B}_{reward}} C_b
  $$

- The reward is normalized across arms that have already been measured. The policy does not compare raw absolute milliseconds directly. For an observed arm `a`:

  $$
  reward(a)
  =
  1
  -
  \frac{\overline{C}_a - C_{min}}{C_{max} - C_{min}}
  $$

  where:

  $$
  C_{min} = \min_{a \in \mathcal{A}_{obs}} \overline{C}_a
  $$

  $$
  C_{max} = \max_{a \in \mathcal{A}_{obs}} \overline{C}_a
  $$

  and \(\overline{C}_a\) is the current mean measured cost of arm \(a\).

  Symbol details:

  $$
  a
  $$

  is one candidate arm, for example \((5, 15)\).

  $$
  \mathcal{A}_{obs}
  $$

  is the set of arms that already have at least one completed six-batch measurement window.

  $$
  \overline{C}_a
  $$

  is the running mean cost of arm \(a\). If the same arm is tested multiple times, each six-batch window produces one measured cost, and \(\overline{C}_a\) is the average of those measured costs.

  $$
  C_{min}
  $$

  is the smallest \(\overline{C}_a\) among all observed arms, meaning the best measured arm so far.

  $$
  C_{max}
  $$

  is the largest \(\overline{C}_a\) among all observed arms, meaning the worst measured arm so far.

  Interpretation:

  ```text
  reward ~= 1.0 means this arm is currently the best observed arm
  reward ~= 0.0 means this arm is currently the worst observed arm
  reward = 0.5 is used when only one arm has been measured or all observed costs are equal
  ```

- The arm selector uses a simple UCB-style score:

  $$
  score(a)
  =
  reward(a)
  +
  c
  \sqrt{
    \frac{\log(N)}{n_a}
  }
  $$

  with:

  $$
  c = 0.5
  $$

  where \(N\) is the total number of measured arm windows, and \(n_a\) is the number of measured windows for arm \(a\).

  Arms that have never been tested are selected first. After every candidate has at least one measured window, the policy balances exploitation of high-reward arms and exploration of less-tested arms.

- Runtime flow on Rank 0 is now:

  ```text
  batch N finishes
      -> collect_batch_summary(batch=N)
      -> run_bandit_after_batch(batch=N)
      -> reallocate_layer(batch=N+1, arm=next_arm)
      -> scheduler.csv stores the layer split for batch N+1
  ```

- `bandit_logs/scheduler_summary_YYYY-MM-DD-HH-MM.csv` remains an audit and post-processing file. The bandit does not read it during online scheduling. Online decisions use `Scheduler.batch_summary_history` directly from memory.

## 2026-07-08

### online data collection

- Added the first minimal online data collection path inside `Scheduler`. The purpose is to let future scheduling algorithms read one compact batch-level signal without parsing the human experiment log.
- The design intentionally follows the same timing value shown in `--- summary after batch ...`. It does not collect detailed per-field records yet. The collected data is limited to:

  ```text
  Batch Number
  Way of layer allocation
  Rank
  One mode-dependent summary time
  ```

- In memory, `Scheduler` now keeps:

  ```text
  scheduler.batch_summary_history[batch]
  ```

  Each entry has this shape:

  ```text
  {
      "batch": batch_number,
      "prefill_mode": "distributed" or "cloud-base",
      "layer_allocation": "rank0=[0,5) rank1=[5,15) rank2=[15,22)",
      "time_label": "...",
      "rank_times": {
          0: time_ms,
          1: time_ms,
          2: time_ms,
      },
  }
  ```

- The same data is written after each completed batch to:

  ```text
  scheduler_summary.csv
  ```

  The file format is intentionally narrow:

  ```text
  batch,prefill_mode,layer_allocation,rank,time_label,time_ms
  ```

- Timing rule for `PREFILL_MODE=distributed`:

  ```text
  time_label = T_comp + T_transfer + T_comm
  time_ms =
      prefill_comp_time_ms
    + prefill_transfer_time_ms
    + decode_comp_time_ms
    + decode_transfer_time_ms
  ```

- Timing rule for `PREFILL_MODE=cloud-base`:

  ```text
  time_label = T_decode + T_transfer + T_comm
  time_ms =
      cloud_prefill_rank2_time_ms
    + kv_cache_send_time_ms
    + kv_cache_recv_time_ms
    + decode_comp_time_ms
    + decode_transfer_time_ms
  ```

- No extra detailed fields are collected in this version. In particular, scheduler online data does not store parameter size, KV-cache size, decode step count, Environment bandwidth, Environment delay, or every raw metric component separately. Those can be added later only if the scheduling policy needs them.
- `inference_loops.py` now calls `scheduler.collect_batch_summary(...)` after Rank 0 receives all rank records for a completed batch and before `scheduler.reallocate_layer(...)`. This gives future policies a stable control point:

  ```text
  batch finished -> summary data collected -> scheduler may decide next allocation
  ```

### Forced decode-step mode

- Added the runtime option `--force-decode-steps N`. This option ignores EOS and forces Rank 0 to drive exactly `N` decode forward steps after prefill.
- Added `FORCE_DECODE_STEPS` to `run.sh`. Leave it empty to keep the normal stopping rule. Set it to a number, for example `128`, to pass `--force-decode-steps 128` to the Python program:

  ```bash
  FORCE_DECODE_STEPS=${FORCE_DECODE_STEPS:-128}
  ```

- The forced count is decode-forward count, not total generated-token count. The first token produced directly by prefill is not counted. Therefore `--force-decode-steps 128` means:

  ```text
  first token from prefill + 128 forced decode forwards
  ```

- In forced mode, Rank 0 keeps every row active even if EOS appears. This keeps batch shape and decode work stable for performance experiments.
- Worker ranks do not need their own stopping logic. They continue to be message-driven: Rank 0 sends hidden states for as many forced decode steps as requested, then sends `batch_done`.

## 2026-07-07

### Cloud-base log field correction

- Corrected the `cloud-base` metric semantics after experiment output showed `Rank 0 prefill_transfer_time_ms` as non-zero. That value came from `Rank 0 -> Rank 2` prompt transfer, but this field belongs to the `distributed parameter` section and should only describe the distributed prefill pipeline.
- In `PREFILL_MODE=cloud-base`, all ranks now keep the distributed prefill fields at zero:

  ```text
  prefill_comp_time_ms=0.00
  prefill_transfer_time_ms=0.00
  ```

- `cloud-base` still performs `Rank 0 -> Rank 2` prompt transfer and `Rank 2 -> Rank 0` first-token transfer, but these are not recorded in `distributed parameter`. The cloud-base section remains focused on:

  ```text
  cloud_prefill_rank2_time_ms
  kv_cache_send_time_ms
  kv_cache_recv_time_ms
  ```

- `Common parameter` remains decode-only. It starts after the first token has already been produced by prefill. If a batch naturally stops after that first token, these fields can be zero:

  ```text
  decode_step_count=0
  decode_comp_time_ms=0.00
  decode_transfer_time_ms=0.00
  decode_time_per_token_ms=0.00
  ```

- Updated `experiment_report.py` summary calculation so the cloud-base summary no longer adds `prefill_transfer_time_ms`.
- Updated `experiment_report.py` summary display so only the active `PREFILL_MODE` prints timing values. The inactive mode is kept as a section header but writes `not applicable`, which avoids implying that distributed and cloud-base prefill both ran in the same batch.
- Updated `inference_loops.py` so cloud-base Rank 0 no longer writes prompt transfer time into `prefill_transfer_time_ms`, and cloud-base Rank 2 no longer writes first-token transfer time into `prefill_transfer_time_ms`.
- Updated `readme.md` to document that `distributed parameter` only applies to `PREFILL_MODE=distributed`.

## 2026-07-06

### Experiment log timing rework

- Rebuilt the experiment timing records around the new compute/transfer split. The previous timing fields mixed different meanings: some measured only local forward, while others measured forward plus outgoing send, and the summary accumulated values across batches. That made later Scheduler and multi-arm-bandit work hard to audit.
- Removed the old batch-crossing total summary model. `append_summary_log()` now writes a current-batch snapshot only. Batch 2 no longer includes Batch 1 time, and later batches are not accumulated into any `total_*` field.
- Removed the old timing fields from active Python records:

  ```text
  prefill_time_ms
  prefill_time_total_ms
  decode_time_total_ms
  inference_compute_total_ms
  total_prefill_time_ms
  total_decode_time_ms
  total_inference_compute_time_ms
  ```

- Added the new active timing fields:

  ```text
  prefill_comp_time_ms
  prefill_transfer_time_ms
  cloud_prefill_rank2_time_ms
  kv_cache_send_time_ms
  kv_cache_recv_time_ms
  decode_step_count
  decode_comp_time_ms
  decode_transfer_time_ms
  decode_time_per_token_ms
  ```

- Timing semantics:

  ```text
  *_comp_time_ms
      Measures rank-local model computation. For prefill this means the local
      prefill forward path. For decode this means the local decode forward path
      and token bookkeeping before the outgoing boundary message.

  *_transfer_time_ms
      Measures outgoing communication from this rank. When Environment is active,
      this includes the simulated bandwidth / fixed-delay wait in addition to
      the real send call.

  decode_time_per_token_ms
      Computed as:
      (decode_comp_time_ms + decode_transfer_time_ms) / decode_step_count
  ```

- Distributed prefill mapping:

  ```text
  Rank 0 prefill_comp_time_ms      = layers [rank0_start, rank0_end) forward
  Rank 0 prefill_transfer_time_ms  = hidden states Rank 0 -> Rank 1

  Rank 1 prefill_comp_time_ms      = middle partition forward
  Rank 1 prefill_transfer_time_ms  = hidden states Rank 1 -> Rank 2

  Rank 2 prefill_comp_time_ms      = final partition forward + first-token logits
  Rank 2 prefill_transfer_time_ms  = first token Rank 2 -> Rank 0
  ```

- Cloud-base prefill mapping:

  ```text
  Rank 0 prefill_comp_time_ms           = 0.00
  Rank 0 prefill_transfer_time_ms       = 0.00
  Rank 0 kv_cache_recv_time_ms          = receive + rebuild Rank 0 KV cache

  Rank 1 prefill_comp_time_ms           = 0.00
  Rank 1 prefill_transfer_time_ms       = 0.00
  Rank 1 kv_cache_recv_time_ms          = receive + rebuild Rank 1 KV cache

  Rank 2 prefill_comp_time_ms           = 0.00
  Rank 2 prefill_transfer_time_ms       = 0.00
  Rank 2 cloud_prefill_rank2_time_ms    = full-model prefill on Rank 2
  Rank 2 kv_cache_send_time_ms          = parallel KV cache metadata/payload send
  ```

- `experiment_report.py` was rewritten around the new record schema. It now prints the record in three sections: `distributed parameter`, `Cloud-base parameter`, and `Common parameter`. It also prints the current batch layer allocation and Environment snapshot in the summary.
- `inference_loops.py` was refactored so Rank 0 and worker ranks no longer use one large timer around "compute + send". Each path now records local forward time separately from outgoing communication time.
- `bandwidth_transfer.py` wrappers now return elapsed milliseconds for limited hidden-state, token, and prefill-input sends. This allows the inference loop to record transfer time without reimplementing the Environment delay calculation.
- `kv_cache_transfer.py` now returns elapsed milliseconds from `send_prefill_inputs()`, but cloud-base prompt transfer is intentionally not written to `prefill_transfer_time_ms` because that field belongs to distributed prefill.
- `readme.md` was rewritten to document only the current startup flow and active log fields.

### Environment-based per-link bandwidth and delay simulation

- Replaced the old single `--bandwidth` / `BANDWIDTH` experiment interface with a Rank-0-owned `Environment` class in `environment.py`. Network simulation is no longer configured from `run.sh`; it is configured by editing `environment.py` on Rank 0 and then broadcasting the active snapshot to worker ranks.
- `Environment` now models five real directed links. The index order is fixed because later multi-arm-bandit code will need a stable state vector:

  ```text
                 [3] 0 -> 2
              +--------------+
              |              v
  Rank 0 --[0]--> Rank 1 --[1]--> Rank 2
     ^                              |
     |                              |
     +----------- [2] 2 -> 0 <------+

  Extra cloud-base KV link:

  Rank 2 --[4]--> Rank 1
  ```

  The corresponding arrays are:

  ```text
  Bandwidth[0],       time_comm_delay[0] = Rank 0 -> Rank 1
  Bandwidth[1],       time_comm_delay[1] = Rank 1 -> Rank 2
  Bandwidth[2],       time_comm_delay[2] = Rank 2 -> Rank 0
  Bandwidth[3],       time_comm_delay[3] = Rank 0 -> Rank 2
  Bandwidth[4],       time_comm_delay[4] = Rank 2 -> Rank 1
  ```

- `Bandwidth` uses MB/s, where `1 MB = 1024 * 1024 bytes`. A value of `None` means unlimited bandwidth on that specific link. `time_comm_delay` uses milliseconds and represents fixed one-way communication delay on that specific link.
- Default values are defined at the top of `environment.py`:

  ```python
  DEFAULT_BANDWIDTH = [None, None, None, None, None]
  DEFAULT_TIME_COMM_DELAY = [0.0, 0.0, 0.0, 0.0, 0.0]
  DEFAULT_SCHEDULE = {}
  ```

- The simulated target time for a payload on link `i` is:

  ```text
  bandwidth_seconds = payload_bytes / (Bandwidth[i] * 1024 * 1024)
  delay_seconds = time_comm_delay[i] / 1000
  expected_seconds = bandwidth_seconds + delay_seconds
  extra_sleep = max(0, expected_seconds - real_nccl_elapsed_seconds)
  ```

  This keeps the old "do not punish slow real communication twice" rule. If real NCCL transfer is already slower than the configured target, no extra sleep is added.
- Batch-level environment changes are reserved through `DEFAULT_SCHEDULE`. A key means "from this batch onward":

  ```python
  DEFAULT_SCHEDULE = {
      10: {
          "Bandwidth": [100, 80, 120, 60, 70],
          "time_comm_delay": [1.0, 1.5, 2.0, 5.0, 4.0],
      }
  }
  ```

  In dynamic mode, Rank 0 calls `environment.apply_batch(batch_number)` at the start of each batch, then broadcasts the active environment to Rank 1 and Rank 2. Worker ranks do not apply their local schedule; they trust Rank 0's broadcast snapshot.
- Communication mapping:

  ```text
  distributed / decode:
    Rank 0 -> Rank 1 hidden states use link [0]
    Rank 1 -> Rank 2 hidden states use link [1]
    Rank 2 -> Rank 0 next token uses link [2]

  cloud-base prefill:
    Rank 0 -> Rank 2 input_ids + attention_mask use link [3]
    Rank 2 -> Rank 0 KV cache partition uses link [2]
    Rank 2 -> Rank 1 KV cache partition uses link [4]
  ```

- `bandwidth_transfer.py` remains the wrapper module name, but its behavior is now Environment-based. It checks per-link bandwidth and per-link delay instead of reading one global bandwidth scalar.
- Cloud-base KV cache still uses a `ready -> metadata -> payload -> done` protocol when either `2 -> 0` or `2 -> 1` has simulation enabled. Both receivers use the same limited receive protocol in that case, so the extra `done` message is consumed consistently.
- Token return `Rank 2 -> Rank 0` now also uses Environment delay when link `[2]` is configured. Because token return has no explicit `done` handshake, the delay is applied before sending the token so Rank 0's `recv_token()` actually waits for the simulated one-way delay.
- `Scheduler` now stores the active environment snapshot after each completed batch through `update_environment_data()`. The data is kept in:

  ```text
  scheduler.environment_data
  scheduler.environment_history[batch_number]
  ```

  Each snapshot contains `Bandwidth[0..4]`, `time_comm_delay[0..4]`, the active batch number, and link names. The current `reallocate_layer()` policy still does not change layer allocation; this environment history is reserved as future multi-arm-bandit input.

### Files changed

- `environment.py`: added the new Environment class, five-link index convention, default bandwidth/delay arrays, batch schedule, broadcast serialization, and timing helpers.
- `distributed_env.py`: replaced the old single-value `broadcast_bandwidth()` with `broadcast_environment()`.
- `distributed_tinyllama_inference.py`: initializes Environment, broadcasts the initial snapshot, prints the active environment, and passes it into inference loops.
- `inference_loops.py`: applies and broadcasts Environment per dynamic batch, routes each communication path to the correct link, and forwards environment snapshots to Scheduler.
- `bandwidth_transfer.py`: changed wrappers from global-bandwidth simulation to Environment-based per-link simulation.
- `scheduler.py`: added `environment_data` and `environment_history` for future bandit decisions.
- `config.py`: removed `--bandwidth`.
- `run.sh`: removed `BANDWIDTH`; startup commands are unchanged.
- `readme.md`: replaced the Bandwidth section with Environment usage and the five-link index table.

## 2026-06-30

### Scheduler is now mandatory on Rank 0

- Re-enabled the scheduler as the default Rank 0 layer-allocation mechanism. `run.sh` now defines `SCHEDULER_CSV=${SCHEDULER_CSV:-scheduler.csv}` and passes `--allocation-csv $SCHEDULER_CSV` only in the Rank 0 command.
- No extra enable or rebuild switch was added. The scheduler is now expected to exist as part of normal execution. If `scheduler.csv` does not exist, Rank 0 creates it automatically from the current `SPLIT_LAYERS` / `--split-layers` default partition.
- Rank 0 remains the single source of truth for layer boundaries. Rank 1 and Rank 2 do not pass `--allocation-csv`; they receive boundaries through `broadcast_boundaries()` at the start of every batch. This avoids stale local scheduler files causing workers to choose a different split.
- Missing batches inherit the latest known scheduler allocation. For example, if `scheduler.csv` only contains batch 1, then batch 2 and later continue using the batch 1 split until a later batch row is explicitly recorded.
- Extended `scheduler.py` with rank metric storage fields: `rank_metrics`, `rank0_data`, `rank1_data`, and `rank2_data`. Rank 0 updates these fields after receiving all per-rank metric records for a completed batch.
- Added `update_rank_metrics()` and `update_rank_metrics_from_records()` to keep the latest per-rank data inside the scheduler. These methods are intentionally light-weight and do not change the CSV by themselves.
- Added the reserved method `reallocate_layer()`. In this version it is a no-op policy hook: it returns the latest known allocation for the requested next batch and does not mutate `scheduler.csv`. Future adaptive scheduling can implement real rules here, using `rank0_data`, `rank1_data`, and `rank2_data` as inputs.
- The current batch loop calls the scheduler hook after batch metrics have been collected and before writing the batch log. This gives future reallocation logic a stable place to run: "batch finished -> metrics collected -> scheduler may decide next allocation".

### Files changed

- `run.sh`: added `SCHEDULER_CSV`, included it in the startup echo, and passed `--allocation-csv` only to Rank 0.
- `scheduler.py`: renamed the documentation language from `allocation.csv` ownership to Rank 0 `scheduler.csv` ownership, added rank metric storage, and added the reserved `reallocate_layer()` hook.
- `inference_loops.py`: Rank 0 now forwards completed-batch metric records into the scheduler and calls the reserved reallocation hook for the next batch.
- `config.py`: updated `--allocation-csv` help text to match the Rank 0 scheduler behavior.
- `readme.md`: rewritten as the current run guide and documented the mandatory scheduler behavior.

## 2026-06-17

### Bandwidth simulation design

- Added optional simulated bandwidth control through `--bandwidth` and the `BANDWIDTH` variable in `run.sh`. The unit is MB/s, with `1 MB = 1024 * 1024 bytes`. To simulate a 100 MB/s communication cap, edit the top of `run.sh` to `BANDWIDTH=${BANDWIDTH:-100}`. Startup still uses the normal per-rank `run.sh` commands.
- Rank 0 is the only configuration source for bandwidth. After `dist.init_process_group()`, Rank 0 broadcasts the effective bandwidth value to all workers through `broadcast_bandwidth()` in `distributed_env.py`. A negative broadcast value means unlimited bandwidth. Rank 1 and Rank 2 ignore any stale local CLI value and use the value received from Rank 0.
- The default behavior is intentionally unchanged. If `--bandwidth` is omitted, `args.bandwidth` remains `None`, and the program calls the original helpers in `pipeline_comm.py` and `kv_cache_transfer.py` directly. This is important for baseline experiments: no extra wrapper, no extra sleep, and no extra `done` message is introduced when bandwidth simulation is not requested.
- `bandwidth_transfer.py` is imported lazily inside the limited branches. This means an unlimited baseline run does not enter the new module at startup and keeps the original communication path as isolated as possible.
- Added `bandwidth_transfer.py` as the only module that owns simulated-bandwidth communication. It contains limited wrappers for large tensor payloads: `send_hidden_limited()`, `send_prefill_inputs_limited()`, `recv_prefill_inputs_limited()`, `send_kv_caches_parallel_limited()`, and `recv_kv_cache_limited()`. Keeping these wrappers in a separate file avoids mixing experiment simulation logic into the original NCCL protocol helpers.
- The limiter does not use a fixed sleep time. It first computes the target transfer time from payload size:

  ```text
  target_seconds = payload_bytes / (bandwidth_MBps * 1024 * 1024)
  ```

  Then it measures the real NCCL send/wait time and only sleeps for the remaining part:

  ```text
  extra_sleep = max(0, target_seconds - real_elapsed_seconds)
  ```

  This means slow real communication is not punished twice. If the actual NCCL transfer is already slower than the simulated bandwidth, no artificial sleep is added.
- For cloud-base KV cache transfer, Rank 2 sends cache partitions to Rank 0 and Rank 1 concurrently. The simulated target wall time therefore follows the slower branch, not the sum of both branches:

  ```text
  target_payload_bytes = max(rank0_cache_payload_bytes, rank1_cache_payload_bytes)
  target_seconds = target_payload_bytes / bandwidth
  ```

  This matches the existing parallel-send design, where wall-clock time should be close to the slowest destination branch.
- The original cloud-base KV transfer protocol remains `ready -> metadata -> payload` when bandwidth is unlimited. In bandwidth mode only, the protocol becomes `ready -> metadata -> payload -> done`. The extra `done` signal is sent after the real payload transfer and any artificial sleep are complete. Rank 0 and Rank 1 wait for this `done` signal inside `recv_kv_cache_limited()`, so their `kv_cache_recv_time_ms` includes the same simulated transfer window that Rank 2 reports in `kv_cache_send_time_ms`.
- In the limited KV-cache receive path, tensors are first received on the communication device, then the receiver waits for `done`, and only after that converts tensors to the compute device/dtype and rebuilds `DynamicCache`. This avoids making Rank 2's `kv_cache_send_time_ms` include Rank 0/1 cache reconstruction time while still keeping Rank 0/1 `kv_cache_recv_time_ms` equal to receive + bandwidth wait + local reconstruction.
- Rank 0 -> Rank 2 cloud-base prefill inputs also use a limited-only `done` signal. Without it, Rank 0 could sleep after sending while Rank 2 already starts full-model prefill, allowing communication delay to overlap with compute and under-report the effect of a low bandwidth cap.
- Small control messages are not bandwidth-limited. This includes status messages, stop messages, batch-done messages, token return messages, boundary broadcasts, prefill-mode broadcasts, bandwidth broadcasts, KV-cache ready signals, KV metadata, and transfer done signals. The simulation is applied only to large tensor payloads where bandwidth has meaningful impact: hidden states, prefill input tensors, and KV-cache key/value tensors.
- The first version is an application-level simulation, not an operating-system or NIC-level traffic shaper. It preserves tensor send/recv semantics and adds timing behavior around the communication calls. This is lower risk for NCCL correctness and keeps future extension straightforward.
- The design reserves a future extension point for dynamic bandwidth. Today `--bandwidth` is a fixed scalar broadcast by Rank 0. Later, `bandwidth_transfer.py` can replace the fixed value with a schedule provider based on batch number, token index, source rank, destination rank, or communication phase without changing the original unlimited communication path.

### Files changed

- `config.py`: added `--bandwidth`, validated it as a positive float when provided.
- `distributed_env.py`: added `broadcast_bandwidth()` so Rank 0 owns the bandwidth setting and workers receive it at startup.
- `distributed_tinyllama_inference.py`: broadcasts bandwidth after prefill mode and prints the effective setting on every rank.
- `bandwidth_transfer.py`: added the bandwidth-limited wrappers and the limited KV-cache `done` protocol.
- `inference_loops.py`: added conditional branches. When `args.bandwidth is None`, it calls the original communication helpers. When a bandwidth value is present, it calls the limited wrappers from `bandwidth_transfer.py`.
- `run.sh`: added optional `BANDWIDTH`. Only Rank 0 passes `--bandwidth`; Rank 1 and Rank 2 receive it through the runtime broadcast.
- `readme.md`: documented `BANDWIDTH` usage and added `bandwidth_transfer.py` to the list of files that must be synchronized across all nodes.

## 2026-06-15

- Changed cloud-base `kv_cache_recv_time_ms` timing semantics. Rank 2 now sends a small ready signal to Rank 0 and Rank 1 immediately before KV-cache metadata/payload transfer. Receivers wait for that ready signal outside the timer, then measure only metadata receive, key/value tensor receive, device/dtype conversion, contiguous normalization, and DynamicCache rebuild.
- Because the KV-cache transfer protocol now includes a ready signal, `kv_cache_transfer.py` must be synchronized to all ranks before running cloud-base mode.

## 2026-06-12

- Changed `PREFILL_MODE` ownership: Rank 0 is now the only rank that decides `distributed` vs `cloud-base`. After NCCL initialization, Rank 0 broadcasts the effective mode to Rank 1 and Rank 2, and worker ranks overwrite their local default before entering any batch loop.
- Updated `run.sh` so only the Rank 0 command passes `--prefill-mode`. Rank 1 and Rank 2 receive the mode from Rank 0 at runtime, which prevents a worker from accidentally entering a different prefill path because of stale local script settings.
- Added cloud-base progress logs around the blocking points: Rank 0 sending tokenized prompts and waiting for KV cache, Rank 1 waiting for KV cache, and Rank 2 waiting for prompt input, running full-model prefill, and sending KV cache partitions. These logs are intended to identify whether a hang is in mode broadcast, prompt transfer, full prefill, KV transfer, or decode.
- Fixed cloud-base decode with transferred KV cache by passing `cache_position` explicitly into `model.model.forward`. Without this, Transformers can infer a stale KV target length from the externally rebuilt cache, producing attention-mask shape errors such as target `[1, 32, 1, 1]` vs mask `[1, 1, 1, 17]`.
- Fixed the transferred-cache layer-index alignment: after pruning model layers for each rank, decoder `layer_idx` values are now renumbered to local indices. Cloud-base sends compact rank-local KV caches, so Rank 1's local layers must address cache slots `[0, local_layer_count)` instead of their original global layer ids.
- Enabled Python `faulthandler` at startup so future native segfaults can still print the Python stack that was active when the process crashed.
- Changed token return path from chained return to direct return. Hidden states still flow `Rank 0 -> Rank 1 -> Rank 2`, but generated tokens now flow directly `Rank 2 -> Rank 0`.
- Rank 1 no longer receives or forwards generated tokens; it only forwards hidden states and keeps the existing metric forwarding path.
- Reworked `run.sh` as the main experiment entry point. Batch size, split layers, init method, model path, dataset path, and Rank 0 CPU/GPU compute mode are now configured at the top of the script.
- Replaced `readme.md` with a concise current-run guide focused on `run.sh`, direct token return, CPU compute, input/output files, logs, and operational notes.
- Added the first cloud-base KV-cache prefill design. `PREFILL_MODE` is kept as the keyword, but its values are now `distributed` and `cloud-base`; the old `pipeline` value is avoided because it conflicts with pipeline parallelism terminology.
- `distributed` keeps the current behavior: each rank computes the KV cache for its own layer partition. `cloud-base` sends tokenized prompts from Rank 0 to Rank 2, lets Rank 2 compute full-model KV cache, then transfers rank-local KV-cache partitions to Rank 0 and Rank 1.
- Added `kv_cache_transfer.py` to isolate cloud-base cache transfer logic from normal pipeline communication. `pipeline_comm.py` continues to own simple hidden/token/status/boundary messages; `kv_cache_transfer.py` owns DynamicCache layer extraction, layer-range splitting, metadata transfer, key/value payload transfer, cache reconstruction, and transfer timing.
- Cloud-base KV-cache transfer uses parallel metadata sends followed by parallel payload sends: metadata is sent with `isend()` to Rank 0 and Rank 1, waited on, then all per-layer key/value tensors are sent with `isend()` and waited on together.
- Extended experiment logs with only the requested cloud-base timing fields: `cloud_prefill_rank2_time_ms`, `kv_cache_send_time_ms`, and `kv_cache_recv_time_ms`. In `cloud-base`, Rank 0/1 prefill time is written as 0 and their cache receive time is recorded separately; Rank 2 records full-model prefill time and parallel cache-send wall-clock time.

## 2026-06-08

- Added optional Rank 0 CPU compute mode through `--compute-device cpu`. Rank 0 can run model forward/KV cache on CPU while moving hidden states and tokens across the existing CUDA/NCCL communication path.
- Updated `run.sh`: `COMPUTE_DEVICE=${COMPUTE_DEVICE:-cuda}` controls Rank 0 compute mode. To use CPU compute on Rank 0, edit the top of `run.sh` to `COMPUTE_DEVICE=${COMPUTE_DEVICE:-cpu}`. Startup commands remain the normal per-rank `run.sh` commands.
- Kept the default behavior unchanged: without `--compute-device cpu`, Rank 0 continues to use GPU compute.
- Extended experiment logs with per-rank decode totals: `decode_step_count`, `decode_time_total_ms`, `prefill_time_total_ms`, and `inference_compute_total_ms`.
- Added a cumulative `summary after batch N` block after every completed batch, so partial runs still preserve per-rank total prefill time, total decode time, total compute time, and total decode step count.

## 2026-05-27

- Changed `decode_time_per_token_ms` from end-to-end token latency to per-rank decode-stage processing time. Rank 0 measures from having the returned token available to sending the next hidden state; middle ranks measure from received hidden state to sent hidden state; the last rank measures from received hidden state to generated logits/token.
- Experiment logs are now flushed and fsynced after every completed batch, and Rank 0 prints the log path after each batch write.

## 2026-05-26

- 默认分布式初始化地址改为 `tcp://10.50.1.228:29500`。
- 实验日志改为写入 `logs/` 文件夹，文件名格式为 `log_YYYY-MM-DD-HH-MM.txt`。
- 新增 `start_three_node_tmux.sh`，用于在本机启动 Rank 0，并通过 SSH 在 Rank 1 / Rank 2 远端节点的 tmux 窗口中启动服务。
- 文档分工调整：`readme.md` 只保留当前运行命令、使用说明和简单更新说明；详细版本改动统一记录在 `log.md`。
- 删除 `start_three_node_tmux.sh`，暂时回到三台机器分别手动启动 Rank 0 / Rank 1 / Rank 2 的方式，降低启动链路复杂度。
- 修复 `batch_size > 1` 时 Rank 1/Rank 2 进入 SDPA attention 可能出现的 `last dimension must be contiguous`：跨 rank 收发的 hidden states、attention mask，以及传入 `model.model()` 的 input/mask/position ids 都会先转为 contiguous。
- 进一步规避同类 SDPA 内部 mask-layout 问题：模型加载时强制 `attn_implementation=eager`，避免 Transformers 在 `batch_size > 1` 时进入 `scaled_dot_product_attention()` 的非连续 bias 路径。
- 将上述 eager attention 兜底改回更直接的 shape 修复：进入 `model.model()` 前显式检查 `[B,Q,H]` / `[B,K]` / `[B,Q]` 的一致性，并由项目代码构造连续的 4D attention bias `[B,1,Q,K]`，避免 Transformers 内部从 2D mask 生成非连续 SDPA bias。
- 根据单节点 `inspect_kv_cache.py` 输出，确认当前 Transformers 返回 `DynamicCache.layers[i].keys / values`。`kv_cache_utils.py` 已补充该结构的统计路径，`kv_cache_size_mb_after_prefill` 后续应能显示非零值。
