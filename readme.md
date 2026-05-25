# Distributed NCCL Inference 测试说明

这个目录用于记录两个 PyTorch 分布式推理测试脚本的使用方法：

- `test_dist1.py`：使用 `gloo` 后端，在 CPU 上测试两个进程之间的分布式通信和简单流水线推理。
- `test_dist_nccl.py`：使用 `nccl` 后端，在 GPU/CUDA 上测试两个节点之间的分布式通信和简单流水线推理。

当前成功测试环境使用的是 PyTorch CUDA 12.1 版本，也就是 `cu121` 版本。

## 1. 脚本作用说明

### `test_dist1.py`

这个脚本使用：

```python
backend="gloo"
init_method="tcp://10.50.0.57:29500"
```

它主要用于验证基础分布式通信是否正常。Rank 0 创建第一段模型 `Linear(4, 8)`，生成输入并计算 hidden states，然后把 hidden shape 和 hidden tensor 发送给 Rank 1。Rank 1 接收 hidden states 后，继续执行第二段模型 `ReLU + Linear(8, 2)`，最后输出结果。

因为它使用 `gloo` 后端，tensor 默认在 CPU 上，不依赖 NCCL 和 GPU 通信。

### `test_dist_nccl.py`

这个脚本使用：

```python
backend="nccl"
init_method="tcp://10.50.0.57:29500"
```

它和 `test_dist1.py` 的推理流程相同，但是所有模型和 tensor 都放在 `cuda:0` 上，并通过 NCCL 后端完成 GPU 之间的通信。

关键代码包括：

```python
device = torch.device("cuda:0")
torch.cuda.set_device(device)
```

因此，每个参与运行的节点都需要可用的 NVIDIA GPU，并且 PyTorch 需要安装支持 CUDA 的版本。

## 2. 前置条件

两台机器都需要满足以下条件：

1. 已安装 NVIDIA 驱动，并且 `nvidia-smi` 可以正常显示 GPU。
2. 已安装 PyTorch CUDA 12.1 版本，例如 `torch` 对应 `cu121`。
3. 两台机器之间网络可以互通。
4. Rank 1 节点可以访问 Rank 0 节点的 `10.50.0.57:29500`。
5. 防火墙没有阻止 `29500` 端口。
6. 两台机器上都存在同一个测试脚本，例如 `test_dist_nccl.py`。

可以用下面命令简单检查 PyTorch 和 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

如果输出中 `torch.cuda.is_available()` 是 `True`，并且 CUDA 版本为 `12.1` 或兼容版本，就可以继续测试 NCCL。

## 3. 网络和 Rank 配置

两个脚本都使用下面这个初始化地址：

```python
init_method="tcp://10.50.0.57:29500"
```

这表示 Rank 0 所在机器的通信地址是：

```text
10.50.0.57:29500
```

如果 Rank 0 机器的 IP 发生变化，需要同时修改两个脚本中的 `init_method`。

本测试使用两个进程，因此：

```bash
WORLD_SIZE=2
```

其中：

- Rank 0：第一台机器，负责生成输入、执行第一段模型，并发送 hidden states。
- Rank 1：第二台机器，负责接收 hidden states、执行第二段模型，并输出最终结果。

## 4. 运行 `test_dist_nccl.py`

### Rank 0 机器运行命令

在 Rank 0 机器上执行：

```bash
export RANK=0
export WORLD_SIZE=2

export NCCL_SOCKET_IFNAME=enp6s18
export NCCL_DEBUG=INFO

python test_dist_nccl.py
```

说明：

- `RANK=0` 表示当前进程是第 0 个分布式进程。
- `WORLD_SIZE=2` 表示总共有 2 个分布式进程。
- `NCCL_SOCKET_IFNAME=enp6s18` 指定 Rank 0 使用的网卡。
- `NCCL_DEBUG=INFO` 打印 NCCL 详细日志，方便排查网络和通信问题。

### Rank 1 机器运行命令

在 Rank 1 机器上执行：

```bash
export RANK=1
export WORLD_SIZE=2

export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

python test_dist_nccl.py
```

说明：

- `RANK=1` 表示当前进程是第 1 个分布式进程。
- `NCCL_SOCKET_IFNAME=eth0` 指定 Rank 1 使用的网卡。
- Rank 1 会连接 Rank 0 的 `10.50.0.57:29500`。

建议先启动 Rank 0，再启动 Rank 1。两个进程都启动后，分布式进程组才会完成初始化。

## 5. 成功运行时的现象

Rank 0 会输出类似内容：

```text
[Rank 0] Starting...
[Rank 0] Hostname: ...

[Rank 0] Input:
...

[Rank 0] Hidden:
...

[Rank 0] Hidden sent

[Rank 0] SUCCESS
```

Rank 1 会输出类似内容：

```text
[Rank 1] Starting...
[Rank 1] Hostname: ...

[Rank 1] Hidden received:
...

[Rank 1] Output:
...

[Rank 1] SUCCESS
```

如果两个进程都打印出 `SUCCESS`，说明 NCCL 分布式通信和简单的跨节点流水线推理已经成功。

## 6. 运行 `test_dist1.py`

如果想先验证基础 TCP 分布式通信，可以运行 CPU/Gloo 版本：

Rank 0：

```bash
export RANK=0
export WORLD_SIZE=2
python test_dist1.py
```

Rank 1：

```bash
export RANK=1
export WORLD_SIZE=2
python test_dist1.py
```

`test_dist1.py` 不需要设置 `NCCL_SOCKET_IFNAME`，因为它使用的是 `gloo` 后端。

如果 Gloo 版本成功，但 NCCL 版本失败，通常说明 Python 分布式初始化和基础网络是通的，问题更可能出在 GPU、NCCL、网卡选择、防火墙或 CUDA/PyTorch 版本兼容性上。

## 7. 常见问题排查

### 1. 卡在 `init_process_group`

可能原因：

- Rank 0 没有先启动。
- Rank 1 无法访问 `10.50.0.57:29500`。
- 防火墙阻止了端口 `29500`。
- `WORLD_SIZE` 设置不一致。
- 两台机器的 `RANK` 设置重复或错误。

检查方式：

```bash
ping 10.50.0.57
```

也可以检查端口连通性：

```bash
nc -vz 10.50.0.57 29500
```

### 2. NCCL 报网卡相关错误

确认网卡名称是否正确：

```bash
ip addr
```

Rank 0 当前使用：

```bash
export NCCL_SOCKET_IFNAME=enp6s18
```

Rank 1 当前使用：

```bash
export NCCL_SOCKET_IFNAME=eth0
```

如果机器网卡名称变化，需要改成对应机器上实际存在且能互通的网卡。

### 3. CUDA 不可用

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

如果 `torch.cuda.is_available()` 为 `False`，需要检查 NVIDIA 驱动、CUDA 运行环境和 PyTorch 安装版本。

### 4. PyTorch 版本不匹配

建议两台机器使用相同的 PyTorch 版本和 CUDA 构建版本。当前测试成功环境为 PyTorch `cu121`。

查看版本：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

### 5. 端口被占用

如果 `29500` 已经被其他进程占用，可以修改两个脚本中的端口，例如改为：

```python
init_method="tcp://10.50.0.57:29501"
```

注意两台机器的脚本必须保持一致。

## 8. 推荐测试顺序

建议按下面顺序测试：

1. 先确认两台机器可以互相 ping 通。
2. 再运行 `test_dist1.py`，确认 Gloo/CPU 分布式通信正常。
3. 确认两台机器 `torch.cuda.is_available()` 都是 `True`。
4. 设置正确的 `NCCL_SOCKET_IFNAME`。
5. 运行 `test_dist_nccl.py`。
6. 看到两个 Rank 都输出 `SUCCESS` 后，说明测试通过。

## 9. 文件路径建议

可以把两个测试脚本放在同一个目录下，例如：

```text
~/distributed-nccl-inference/
├── markdown.md
├── test_dist1.py
└── test_dist_nccl.py
```

然后分别在两台机器进入该目录执行对应命令。

## 10. TinyLlama 双节点分层联合推理

当前项目新增 `distributed_tinyllama_inference.py`，用于把本地 TinyLlama 按 decoder layer 切到两个节点上进行联合推理。

默认切分位置是第 5 层：

- Rank 0：负责 tokenizer、embedding、前 5 个 decoder layer，也就是 layer `0` 到 layer `4`。
- Rank 1：负责从第 5 层开始的剩余 decoder layer、final norm 和 `lm_head`。
- Rank 0 会把 hidden states 通过 NCCL 发送给 Rank 1。
- Rank 1 计算 next token 后，再通过 NCCL 把 token 发回 Rank 0。
- Rank 0 继续拼接 token，并逐 token 生成最终输出。

### 模型目录

请在项目目录下创建模型目录，并把 TinyLlama 放进去：

```text
C:\Users\smbu\Desktop\distributed-nccl-inference\model\tinyllama
```

目录里应该包含 Hugging Face 格式模型文件，例如：

```text
config.json
tokenizer.json
tokenizer.model
model.safetensors
```

如果模型是分片保存，也可以是多个 `model-0000x-of-0000x.safetensors` 文件。

### CSV 输入格式

默认输入文件是：

```text
prompts.csv
```

一行一条 prompt。当前脚本支持两种格式。

有表头时：

```csv
prompt
Please introduce TinyLlama in one short paragraph.
Write a short checklist for debugging NCCL communication.
```

运行时加上：

```bash
--csv-has-header --prompt-column prompt
```

没有表头时：

```csv
Please introduce TinyLlama in one short paragraph.
Write a short checklist for debugging NCCL communication.
```

默认读取第一列。

### Rank 0 运行命令

在 master 节点进入项目目录后执行：

```bash
cd ~/distributed-nccl-inference

export RANK=0
export WORLD_SIZE=2
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

python distributed_tinyllama_inference.py \
  --model-dir model/tinyllama \
  --input-csv prompts.csv \
  --output-csv outputs.csv \
  --csv-has-header \
  --prompt-column prompt \
  --max-new-tokens 64
```

如果 master 地址不是脚本默认的 `tcp://ksai.scnet.cn:29500`，可以显式指定：

```bash
python distributed_tinyllama_inference.py \
  --init-method tcp://10.50.0.57:29500 \
  --model-dir model/tinyllama \
  --input-csv prompts.csv \
  --output-csv outputs.csv \
  --csv-has-header \
  --prompt-column prompt
```

### Rank 1 运行命令

在 slave 节点进入相同项目目录后执行：

```bash
cd ~/distributed-nccl-inference

export RANK=1
export WORLD_SIZE=2
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

python distributed_tinyllama_inference.py \
  --model-dir model/tinyllama \
  --csv-has-header \
  --prompt-column prompt \
  --max-new-tokens 64
```

Rank 1 不会读取 CSV，也不会写输出文件；它只接收 Rank 0 发来的 hidden states，继续跑剩余模型层，并把 next token 发回 Rank 0。

### 输出文件

Rank 0 默认生成：

```text
outputs.csv
```

输出列包括：

- `prompt`：原始输入。
- `generated_text`：模型新生成的文本。
- `full_text`：prompt 加生成文本的完整结果。

### 常用参数

```bash
--split-layer 5
```

指定切分位置。默认是 5，也就是 Rank 1 从第 5 层开始执行。

```bash
--max-new-tokens 64
```

每条 prompt 最多生成多少个新 token。

```bash
--temperature 0
```

默认使用贪心解码。设置为大于 0 的值时启用采样，例如 `--temperature 0.7`。

```bash
--dtype float16
```

默认使用 `float16`。如果 GPU 支持 BF16，也可以使用 `--dtype bfloat16`。

### 推荐测试顺序

1. 先确认原来的 `test_dist_nccl.py` 仍然可以双节点成功。
2. 确认两台机器上都存在相同的 `model/tinyllama` 目录。
3. 在 Rank 0 准备 `prompts.csv`。
4. 先启动 Rank 0，再启动 Rank 1。
5. Rank 0 生成完成后检查 `outputs.csv`。

## 11. 2026-05-19 版本更新：按节点懒加载 TinyLlama 权重

本版本在 `distributed_tinyllama_inference.py` 中新增 `--lazy-load` 参数，用于解决之前两个节点都会完整读取 checkpoint 的问题。

### 改动背景

旧版本的加载流程是：

```text
两个 Rank 都调用 AutoModelForCausalLM.from_pretrained()
两个 Rank 都完整读取 TinyLlama checkpoint
再按 split_layer 删除本节点不执行的 decoder layers
```

因此即使 Rank 0 只执行前 5 层、Rank 1 只执行后 17 层，日志里仍然会看到两个节点都在加载完整权重，例如 `201/201`。

新版本增加了选择性加载路径：

```text
读取 config.json 构建模型结构
按 Rank 先裁剪模型模块
读取 safetensors index
只从 checkpoint 中读取当前 Rank 需要的 tensor
```

### 默认行为保持不变

为了保留已经验证成功的稳定路径，默认命令仍然使用完整加载：

```bash
python distributed_tinyllama_inference.py \
  --model-dir /home/dingcong/models/TinyLlama \
  --input-csv prompts.csv \
  --output-csv outputs.csv \
  --csv-has-header \
  --prompt-column prompt
```

默认路径会打印：

```text
load_mode=full
```

### 启用懒加载

在 Rank 0 和 Rank 1 两边都加上：

```bash
--lazy-load
```

Rank 0 示例：

```bash
export RANK=0
export WORLD_SIZE=2
export NCCL_SOCKET_IFNAME=enp6s18
export NCCL_DEBUG=INFO

python distributed_tinyllama_inference.py \
  --lazy-load \
  --init-method tcp://10.50.0.57:29500 \
  --model-dir /home/dingcong/models/TinyLlama \
  --input-csv prompts.csv \
  --output-csv outputs.csv \
  --csv-has-header \
  --prompt-column prompt
```

Rank 1 示例：

```bash
export RANK=1
export WORLD_SIZE=2
export NCCL_SOCKET_IFNAME=ens12f1np1
export NCCL_DEBUG=INFO

python distributed_tinyllama_inference.py \
  --lazy-load \
  --init-method tcp://10.50.0.57:29500 \
  --model-dir /data-store/pengying/dingcong/models/TinyLlama \
  --csv-has-header \
  --prompt-column prompt
```

懒加载成功时会看到类似输出：

```text
[Rank 0] Lazy-loaded xx tensors from x safetensors shard(s).
[Rank 0] Loaded TinyLlama from ...; total_layers=22; split_layer=5; load_mode=lazy

[Rank 1] Lazy-loaded xx tensors from x safetensors shard(s).
[Rank 1] Loaded TinyLlama from ...; total_layers=22; split_layer=5; load_mode=lazy
```

### 当前权重分配

默认 `--split-layer 5` 时：

- Rank 0 加载并执行 `model.embed_tokens` 和 `model.layers.0` 到 `model.layers.4`。
- Rank 1 加载并执行 `model.layers.5` 到最后一层、`model.norm` 和 `lm_head`。
- Rank 0 不再加载 `lm_head` 和后半部分 decoder layer。
- Rank 1 不再加载 token embedding 和前 5 层 decoder layer。

### 重要限制

`--lazy-load` 当前要求模型是 Hugging Face `safetensors` 格式。模型目录中至少需要存在以下文件之一：

```text
model.safetensors
model.safetensors.index.json
```

如果模型是 `pytorch_model.bin` 格式，建议先继续使用默认完整加载路径，或者把模型转换为 safetensors 后再使用 `--lazy-load`。

### 回退方式

如果懒加载运行中遇到 checkpoint key 不匹配、safetensors 缺失、模型结构差异等问题，直接去掉：

```bash
--lazy-load
```

即可回到之前已经验证通过的完整加载模式。

## 12. 2026-05-20 版本更新：动态加载与 Scheduler 批调度

本版本新增动态加载功能，用于 prompt 数量很多时按 batch 推理，并在每个 batch 开始前检查两台节点各自负责的模型层范围是否发生变化。

### 新增文件

新增：

```text
scheduler.py
```

其中包含 `Scheduler` 类，负责维护：

```text
allocation.csv
```

`allocation.csv` 的格式为：

```csv
batch,rank0,rank1
1,"[0,5)","[5,22)"
2,"[0,5)","[5,22)"
```

含义：

- `batch`：第几个 batch，当前实现从 1 开始计数。
- `rank0`：Rank 0 负责的 decoder layer 区间，格式为 `[start,end)`。
- `rank1`：Rank 1 负责的 decoder layer 区间，格式为 `[start,end)`。

例如：

```csv
batch,rank0,rank1
123,"[0,8)","[8,22)"
```

表示从 batch 123 开始：

- Rank 0 执行 layer `0` 到 layer `7`。
- Rank 1 执行 layer `8` 到最后一层。

### 动态加载的工作方式

开启动态加载后，Rank 0 会：

1. 读取 CSV 中的所有 prompt。
2. 按 `--batch-size` 分成多个真实 prompt tensor batch。
3. 每个 batch 开始前询问 `Scheduler` 当前 batch 的 layer 分配。
4. 把当前 batch 的 midpoint 广播给 Rank 1。
5. Rank 0 和 Rank 1 对比当前已加载模型分区和新的 midpoint。
6. 如果 midpoint 没变，继续复用已加载的模型分区。
7. 如果 midpoint 变化，才释放旧分区并懒加载新的分区。
8. 当前 batch 推理结束后保留模型分区，进入下一个 batch。

Rank 1 不读取 prompt，也不写输出文件。Rank 1 只接收 Rank 0 广播的 midpoint，并把收到的分配记录到本地 `allocation.csv`，便于调试和对照日志。

### 重要要求

动态加载依赖懒加载，因此必须同时启用：

```bash
--lazy-load --dynamic-load
```

如果只写 `--dynamic-load`，脚本会直接报错：

```text
--dynamic-load requires --lazy-load.
```

### Rank 0 运行示例

```bash
export RANK=0
export WORLD_SIZE=2
export NCCL_SOCKET_IFNAME=enp6s18
export NCCL_DEBUG=INFO

python distributed_tinyllama_inference.py \
  --lazy-load \
  --dynamic-load \
  --batch-size 64 \
  --allocation-csv allocation.csv \
  --init-method tcp://10.50.0.57:29500 \
  --model-dir /home/dingcong/models/TinyLlama \
  --input-csv prompts.csv \
  --output-csv outputs.csv \
  --csv-has-header \
  --prompt-column prompt
```

### Rank 1 运行示例

```bash
export RANK=1
export WORLD_SIZE=2
export NCCL_SOCKET_IFNAME=ens12f1np1
export NCCL_DEBUG=INFO

python distributed_tinyllama_inference.py \
  --lazy-load \
  --dynamic-load \
  --batch-size 64 \
  --allocation-csv allocation.csv \
  --init-method tcp://10.50.0.57:29500 \
  --model-dir /data-store/pengying/dingcong/models/TinyLlama \
  --csv-has-header \
  --prompt-column prompt
```

### 手动调整 allocation.csv

如果 `allocation.csv` 已经存在，`Scheduler` 会优先读取里面已有的 batch 分配。

例如可以手动写：

```csv
batch,rank0,rank1
1,"[0,5)","[5,22)"
2,"[0,8)","[8,22)"
3,"[0,10)","[10,22)"
```

这样第 1、2、3 个 batch 会使用不同的层切分位置。没有明确写入的 batch 不会回到默认分配，而是沿用最近一次已有分配。

例如只写：

```csv
batch,rank0,rank1
1,"[0,5)","[5,22)"
20,"[0,8)","[8,22)"
```

则 batch `2` 到 batch `19` 会沿用 batch `1` 的 `[0,5)` / `[5,22)`；batch `21` 之后会沿用 batch `20` 的 `[0,8)` / `[8,22)`。只有在 `allocation.csv` 为空或不存在时，才使用 `--split-layer` 初始化第一条分配。

### 新增参数

```bash
--dynamic-load
```

开启动态加载模式。

```bash
--batch-size 64
```

每个 batch 包含多少条 prompt。默认值是 `64`。对于 1k 到 100k 级别的输入文件，通常可以按机器吞吐和显存情况选择 `32`、`64` 或 `128`。

当前版本中 `--batch-size` 在普通模式和 `--dynamic-load` 模式下都会生效。Rank 0 会把同一个 batch 内的多条 prompt 一次性 tokenizer padding 成张量批，并沿 pipeline 传递对应的 attention mask。Rank 1 / Rank 2 会按相同 batch 维度处理 hidden states，最后一层节点返回形状为 `[batch_size, 1]` 的 next token。

```bash
--allocation-csv allocation.csv
```

指定 Scheduler 维护的分配文件。默认值是 `allocation.csv`。

### 日志现象

动态加载模式下，每个 batch 会看到类似日志：

```text
[Rank 0] Batch 1: rank0=[0,5) rank1=[5,22); prompts=64
[Rank 0] Batch 1 model loaded; midpoint=5; load_mode=lazy
[Rank 0] Batch 1 complete.
[Rank 0] Batch 2: reuse cached model partition for midpoint=5.

[Rank 1] Batch 1: rank0=[0,5) rank1=[5,22)
[Rank 1] Batch 1 model loaded; midpoint=5; load_mode=lazy
[Rank 1] Batch 1 complete.
[Rank 1] Batch 2: reuse cached model partition for midpoint=5.
```

这表示 batch 1 按 `allocation.csv` 完成了模型层分配和加载；batch 2 的分配没有变化，因此直接复用已加载的模型分区。只有当后续 batch 的 midpoint 变化时，脚本才会释放旧分区并重新懒加载。

### 回退方式

如果动态加载过程中需要回到上一版稳定行为，去掉：

```bash
--dynamic-load
```

如果需要回到完整加载模式，同时去掉：

```bash
--lazy-load --dynamic-load
```

## 13. 2026-05-21 版本更新：支持三节点 Pipeline 推理

本版本在保留两节点推理和动态加载功能的基础上，新增三节点 pipeline 推理。

### 三节点职责

三节点时：

```text
Rank 0 / master : tokenizer + embed_tokens + 前段 decoder layers
Rank 1 / node1  : 中段 decoder layers
Rank 2 / node2  : 后段 decoder layers + model.norm + lm_head + next token
```

数据流为：

```text
master <-> node1 <-> node2
```

单步生成时的实际流向：

```text
Rank 0 -> Rank 1 -> Rank 2 -> Rank 1 -> Rank 0
```

Rank 0 不直接向 Rank 2 发送 hidden states，也不直接从 Rank 2 接收 token。Rank 1 是中间 pipeline 节点，负责转发 hidden states 和 next token。

### 两节点启动

两节点方式保持兼容：

```bash
export WORLD_SIZE=2
export RANK=0   # master
python distributed_tinyllama_inference.py ...
```

```bash
export WORLD_SIZE=2
export RANK=1   # slave
python distributed_tinyllama_inference.py ...
```

两节点默认仍然使用：

```text
rank0=[0,5)
rank1=[5,22)
```

也可以显式指定：

```bash
--split-layers 5
```

### 三节点启动

三节点需要分别启动 Rank 0、Rank 1、Rank 2：

```bash
export WORLD_SIZE=3
export RANK=0
python distributed_tinyllama_inference.py \
  --lazy-load \
  --dynamic-load \
  --split-layers 5,15 \
  --allocation-csv allocation.csv \
  --model-dir /home/dingcong/models/TinyLlama \
  --input-csv prompts.csv \
  --output-csv outputs.csv \
  --csv-has-header \
  --prompt-column prompt
```

```bash
export WORLD_SIZE=3
export RANK=1
python distributed_tinyllama_inference.py \
  --lazy-load \
  --dynamic-load \
  --split-layers 5,15 \
  --allocation-csv allocation.csv \
  --model-dir /data-store/pengying/dingcong/models/TinyLlama \
  --csv-has-header \
  --prompt-column prompt
```

```bash
export WORLD_SIZE=3
export RANK=2
python distributed_tinyllama_inference.py \
  --lazy-load \
  --dynamic-load \
  --split-layers 5,15 \
  --allocation-csv allocation.csv \
  --model-dir /data-store/pengying/dingcong/models/TinyLlama \
  --csv-has-header \
  --prompt-column prompt
```

三节点默认切分为：

```text
rank0=[0,5)
rank1=[5,15)
rank2=[15,22)
```

如果不传 `--split-layers`，脚本也会使用上面的默认切分。对于非 TinyLlama 或总层数不同的模型，如果第二个默认切分点不合法，脚本会自动退回到接近三等分的切法。

### 三节点 allocation.csv

三节点时，`allocation.csv` 使用三列 rank 分配：

```csv
batch,rank0,rank1,rank2
1,"[0,5)","[5,15)","[15,22)"
20,"[0,6)","[6,16)","[16,22)"
```

没有明确写入的 batch 继续沿用最近一次已有分配。例如 batch `2` 到 `19` 沿用 batch `1`，batch `21` 之后沿用 batch `20`。

### 新增参数

```bash
--split-layers 5
```

两节点时使用一个 split。

```bash
--split-layers 5,15
```

三节点时使用两个 split。

旧参数仍保留：

```bash
--split-layer 5
```

在两节点模式下它等价于 `--split-layers 5`。在三节点模式下，如果没有显式传 `--split-layers`，它会作为第一个 split，第二个 split 默认使用 `15`。

### 动态加载行为

动态加载仍然保留，并升级为按完整 boundaries 判断是否重载。

两节点 boundaries：

```text
[0,5,22]
```

三节点 boundaries：

```text
[0,5,15,22]
```

每个 rank 只比较自己负责的 `[layer_start, layer_end)` 是否变化：

- 没变：继续复用当前已加载模型分区。
- 变化：释放旧分区，按新区间懒加载。

## 14. 2026-05-21 版本更新：按功能拆分推理代码

本版本将原来较长的 `distributed_tinyllama_inference.py` 按功能拆成多个文件。启动入口仍然不变，继续运行：

```bash
python distributed_tinyllama_inference.py ...
```

### 最新文件结构

```text
distributed_tinyllama_inference.py   # 主入口，只负责启动流程
config.py                            # 参数解析、默认值、层切分、状态码
distributed_env.py                   # RANK/WORLD_SIZE、CUDA device、NCCL 初始化
model_loader.py                      # full load、lazy load、模型裁剪、权重读取
model_forward.py                     # attention mask、position ids、各 rank forward
pipeline_comm.py                     # NCCL send/recv、hidden/token/status/boundaries
csv_io.py                            # prompts.csv 读取、outputs.csv 写入、batch 切分
inference_loops.py                   # Rank 0 推理循环、Rank 1/2 服务循环
scheduler.py                         # 动态加载 allocation.csv 管理
```

### 排查定位

如果启动参数、模型目录、batch size、split layers 不符合预期，先看 `config.py`。

如果进程启动失败、CUDA 设备不对、NCCL 初始化卡住，先看 `distributed_env.py`，同时检查 `RANK`、`WORLD_SIZE`、`NCCL_SOCKET_IFNAME` 和 `--init-method`。

如果懒加载权重缺失、层范围错误、仍然像完整加载一样读取太多权重，先看 `model_loader.py`。

如果 batch 推理、padding、attention mask、position ids、输出 token 质量异常，先看 `model_forward.py`。

如果卡在节点通信、hidden shape 不一致、`batch_size` 跨节点不一致、三节点转发异常，先看 `pipeline_comm.py`。

如果输入 prompt 数量不对、CSV 表头列名不对、输出文件异常，先看 `csv_io.py`。

如果动态加载没有复用模型、batch 切换逻辑异常、scheduler 分配和实际加载不一致，先看 `inference_loops.py` 和 `scheduler.py`。

### 新增 CUDA 设备参数

本版本新增：

```bash
--cuda-device 1
```

默认值仍然是：

```bash
--cuda-device 0
```

如果某台机器的 `cuda:0` 被占用，可以临时指定其他 GPU。例如：

```bash
python distributed_tinyllama_inference.py \
  --cuda-device 1 \
  --lazy-load \
  --dynamic-load \
  --model-dir /home/dingcong/models/TinyLlama
```

也可以继续使用 `CUDA_VISIBLE_DEVICES` 控制进程能看到的 GPU。`--cuda-device` 是当前进程视角下的设备编号。

### 两节点动态加载启动命令

Rank 0：

```bash
export WORLD_SIZE=2
export RANK=0
export NCCL_SOCKET_IFNAME=enp6s18
export NCCL_DEBUG=INFO

python distributed_tinyllama_inference.py \
  --lazy-load \
  --dynamic-load \
  --batch-size 64 \
  --split-layers 5 \
  --allocation-csv allocation.csv \
  --init-method tcp://10.50.0.57:29500 \
  --model-dir /home/dingcong/models/TinyLlama \
  --input-csv prompts.csv \
  --output-csv outputs.csv \
  --csv-has-header \
  --prompt-column prompt \
  --cuda-device 0
```

Rank 1：

```bash
export WORLD_SIZE=2
export RANK=1
export NCCL_SOCKET_IFNAME=ens12f1np1
export NCCL_DEBUG=INFO

python distributed_tinyllama_inference.py \
  --lazy-load \
  --dynamic-load \
  --batch-size 64 \
  --split-layers 5 \
  --allocation-csv allocation.csv \
  --init-method tcp://10.50.0.57:29500 \
  --model-dir /data-store/pengying/dingcong/models/TinyLlama \
  --csv-has-header \
  --prompt-column prompt \
  --cuda-device 0
```

### 三节点动态加载启动命令

Rank 0：

```bash
export WORLD_SIZE=3
export RANK=0
export NCCL_SOCKET_IFNAME=enp6s18
export NCCL_DEBUG=INFO

python distributed_tinyllama_inference.py \
  --lazy-load \
  --dynamic-load \
  --batch-size 64 \
  --split-layers 5,15 \
  --allocation-csv allocation.csv \
  --init-method tcp://10.50.0.57:29500 \
  --model-dir /home/dingcong/models/TinyLlama \
  --input-csv prompts.csv \
  --output-csv outputs.csv \
  --csv-has-header \
  --prompt-column prompt \
  --cuda-device 0
```

Rank 1：

```bash
export WORLD_SIZE=3
export RANK=1
export NCCL_SOCKET_IFNAME=ens12f1np1
export NCCL_DEBUG=INFO

python distributed_tinyllama_inference.py \
  --lazy-load \
  --dynamic-load \
  --batch-size 64 \
  --split-layers 5,15 \
  --allocation-csv allocation.csv \
  --init-method tcp://10.50.0.57:29500 \
  --model-dir /data-store/pengying/dingcong/models/TinyLlama \
  --csv-has-header \
  --prompt-column prompt \
  --cuda-device 0
```

Rank 2：

```bash
export WORLD_SIZE=3
export RANK=2
export NCCL_SOCKET_IFNAME=ens12f1np1
export NCCL_DEBUG=INFO

python distributed_tinyllama_inference.py \
  --lazy-load \
  --dynamic-load \
  --batch-size 64 \
  --split-layers 5,15 \
  --allocation-csv allocation.csv \
  --init-method tcp://10.50.0.57:29500 \
  --model-dir /data-store/pengying/dingcong/models/TinyLlama \
  --csv-has-header \
  --prompt-column prompt \
  --cuda-device 0
```

三节点数据流仍然是：

```text
Rank 0 -> Rank 1 -> Rank 2 -> Rank 1 -> Rank 0
```

Rank 0 负责 tokenizer、embedding、前段 decoder layers、读取输入 CSV、写输出 CSV。Rank 1 负责中间 decoder layers 和转发。Rank 2 负责后段 decoder layers、final norm、lm_head 和 next token 选择。

## 15. 2026-05-25 版本更新：默认 KV Cache 推理与 Prefill 实验日志

本版本将主推理流程升级为默认使用 KV cache。不再保留“每一步重新计算完整序列”的无 cache 主路径。

新的生成流程分为两段：

```text
prefill: 对完整 prompt batch 计算一次，并在每个 rank 本地建立 KV cache
decode : 每一步只计算最新 token，并复用各 rank 自己保存的 KV cache
```

停止规则固定为：

```text
全部 prompt 生成 EOS，或者达到 --max-new-tokens 上限
```

`--max-new-tokens` 默认值改为：

```bash
--max-new-tokens 512
```

### dataset.csv 格式

输入 CSV 格式保持不变。带表头时：

```csv
prompt
这里是一条 prompt
这里是另一条 prompt
```

启动时使用：

```bash
--csv-has-header
--prompt-column prompt
```

如果没有表头，则默认读取第一列，不需要加 `--csv-has-header`。

长 prompt 实验可以使用：

```bash
--max-input-tokens 1000
```

短 prompt 实验可以使用：

```bash
--max-input-tokens 100
```

batch 实验建议分别测试：

```bash
--batch-size 1
--batch-size 16
--batch-size 64
--batch-size 128
```

### split 与 allocation.csv 规则

如果启动命令没有传：

```bash
--allocation-csv allocation.csv
```

则每个 batch 都固定使用 `--split-layers` 指定的层分配。例如三节点：

```bash
--split-layers 5,15
```

表示：

```text
rank0=[0,5)
rank1=[5,15)
rank2=[15,22)
```

只有显式传入：

```bash
--allocation-csv allocation.csv
```

才会启用 Scheduler，并按 `allocation.csv` 对每个 batch 读取或记录层分配。

KV cache 的生命周期是 batch 内有效、batch 间清空重建。因此即使以后使用 `allocation.csv`，也不做跨 batch cache 迁移。

### 实验日志

Rank 0 会自动输出一个文本日志：

```text
log_YYYY-MM-DD-HH-MM.txt
```

例如：

```text
log_2026-05-25-20-45.txt
```

日志文件写在当前运行目录。每个 `--- record ---` 表示一个 batch 内一个 rank 的 prefill 指标。文件格式是一行一个参数，容量单位统一为 MB。

示例：

```text
--- record ---
timestamp=2026-05-25 20:45:12
batch=1
rank=0
world_size=3
layer_start=0
layer_end=5
layer_count=5
batch_size=16
prompt_count=16
input_seq_len_max=1000
input_seq_len_avg=1000.00
dtype=torch.float16
prefill_param_count=123456789
prefill_param_size_mb=235.47
kv_cache_size_mb_after_prefill=78.13
hidden_prefill_shape=[16,1000,2048]
hidden_prefill_size_mb=62.50
cuda_memory_allocated_before_prefill_mb=300.12
cuda_memory_allocated_after_prefill_mb=460.28
cuda_memory_reserved_after_prefill_mb=512.00
prefill_time_ms=123.45
```

字段说明：

- `layer_start/layer_end/layer_count`：当前 rank 分到的 decoder layer 范围。
- `batch_size`：命令行设置的 batch 大小；最后一个 batch 可能更小。
- `prompt_count`：当前 batch 实际 prompt 数量。
- `input_seq_len_max`：当前 batch padding 后的最大输入 token 长度。
- `input_seq_len_avg`：当前 batch 的平均有效输入 token 数。
- `prefill_param_count`：当前 rank 模型分区的参数个数。
- `prefill_param_size_mb`：当前 rank 模型分区参数容量，单位 MB。
- `kv_cache_size_mb_after_prefill`：prefill 完成后当前 rank 的 KV cache 容量，单位 MB。
- `hidden_prefill_shape`：prefill 阶段当前 rank 输出或处理的 hidden states 形状。
- `hidden_prefill_size_mb`：prefill 阶段 hidden states tensor 容量，单位 MB。
- `cuda_memory_allocated_before_prefill_mb`：prefill 前 CUDA allocated memory。
- `cuda_memory_allocated_after_prefill_mb`：prefill 后 CUDA allocated memory。
- `cuda_memory_reserved_after_prefill_mb`：prefill 后 CUDA reserved memory。
- `prefill_time_ms`：当前 rank 本地 prefill forward 计算时间，单位毫秒。

### 三节点固定 split 启动命令

Rank 0：

```bash
export WORLD_SIZE=3
export RANK=0
export NCCL_SOCKET_IFNAME=enp6s18
export NCCL_DEBUG=INFO

python distributed_tinyllama_inference.py \
  --lazy-load \
  --dynamic-load \
  --batch-size 16 \
  --split-layers 5,15 \
  --init-method tcp://10.50.0.57:29500 \
  --model-dir /home/dingcong/models/TinyLlama \
  --input-csv dataset.csv \
  --output-csv outputs_kv.csv \
  --csv-has-header \
  --prompt-column prompt \
  --max-input-tokens 1000
```

Rank 1：

```bash
export WORLD_SIZE=3
export RANK=1
export NCCL_SOCKET_IFNAME=ens12f1np1
export NCCL_DEBUG=INFO

python distributed_tinyllama_inference.py \
  --lazy-load \
  --dynamic-load \
  --batch-size 16 \
  --split-layers 5,15 \
  --init-method tcp://10.50.0.57:29500 \
  --model-dir /data-store/pengying/dingcong/models/TinyLlama \
  --csv-has-header \
  --prompt-column prompt \
  --max-input-tokens 1000
```

Rank 2：

```bash
export WORLD_SIZE=3
export RANK=2
export NCCL_SOCKET_IFNAME=ens12f1np1
export NCCL_DEBUG=INFO

python distributed_tinyllama_inference.py \
  --lazy-load \
  --dynamic-load \
  --batch-size 16 \
  --split-layers 5,15 \
  --init-method tcp://10.50.0.57:29500 \
  --model-dir /data-store/pengying/dingcong/models/TinyLlama \
  --csv-has-header \
  --prompt-column prompt \
  --max-input-tokens 1000
```

如果需要按 `allocation.csv` 进行 batch 级动态分层，在三个 rank 的命令中都显式增加：

```bash
--allocation-csv allocation.csv
```
