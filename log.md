# Version Log

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
