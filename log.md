# Version Log

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
