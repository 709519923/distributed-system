# Version Log

## 2026-05-27

- Changed `decode_time_per_token_ms` from end-to-end token latency to per-rank decode-stage processing time. Rank 0 measures from having the returned token available to sending the next hidden state; middle ranks measure from received hidden state to sent hidden state; the last rank measures from received hidden state to generated logits/token.

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
