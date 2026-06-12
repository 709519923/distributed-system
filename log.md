# Version Log

## 2026-06-12

- Changed token return path from chained return to direct return. Hidden states still flow `Rank 0 -> Rank 1 -> Rank 2`, but generated tokens now flow directly `Rank 2 -> Rank 0`.
- Rank 1 no longer receives or forwards generated tokens; it only forwards hidden states and keeps the existing metric forwarding path.
- Reworked `run.sh` as the main experiment entry point. Batch size, split layers, init method, model path, dataset path, and Rank 0 CPU/GPU compute mode are now configured at the top of the script.
- Replaced `readme.md` with a concise current-run guide focused on `run.sh`, direct token return, CPU compute, input/output files, logs, and operational notes.

## 2026-06-08

- Added optional Rank 0 CPU compute mode through `--compute-device cpu`. Rank 0 can run model forward/KV cache on CPU while moving hidden states and tokens across the existing CUDA/NCCL communication path.
- Updated `run.sh`: `COMPUTE_DEVICE=${COMPUTE_DEVICE:-cuda}` controls Rank 0 compute mode. Use `COMPUTE_DEVICE=cpu bash run.sh 0` when Rank 0 should compute on CPU; Rank 1 and Rank 2 keep the normal GPU commands.
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
