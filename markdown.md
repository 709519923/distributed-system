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
