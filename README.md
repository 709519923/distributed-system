# Lab3 Main Script Guide


## Navigation

This folder contains several main scripts that explore different distributed inference and latency simulation ideas:

### `main.py`
- Baseline script.
- Uses a fixed device map where `model.layers.0` runs on CPU and all later layers run on `cuda:0`.
- Registers a hook on the boundary layer to simulate output transfer latency.
- Runs a short prompt and prints total inference time.

### `main1.py`
- Builds on `main.py` with more robust loading logic.
- Adds error handling for model/tokenizer loading.
- Ensures `pad_token` is set if missing.
- Prints more detailed simulation information and timing.

### `main2.py`
- Extends the idea with dynamic layer slicing.
- Uses `NUM_CPU_LAYERS` to decide how many initial layers stay on CPU.
- Registers the latency hook on the last CPU layer in the current slice.
- Useful for experimenting with different CPU/GPU partition points.

### `main3.py`
- Improves transfer hook reporting.
- Parses layer index from the module if the layer does not expose it directly.
- Prints transfer sizes in KB for easier debugging of small boundary transfers.
- Keeps the same dynamic CPU/GPU slicing behavior.

### `main4.py`
- Simulates a three-stage pipeline with a CPU stage and two logical GPU stages.
- Uses separate bandwidth settings for CPU→GPU0 and GPU0→GPU1 transfers.
- Registers hooks at both pipeline boundaries for multi-hop latency modeling.
- Useful for testing edge-cloud or multi-device partition scenarios.

## How to use

1. Choose one script to run, for example:
   ```bash
   python main1.py
   ```
2. Adjust the configuration constants at the top of the chosen script:
   - `MODEL_ID`
   - `SIMULATED_BANDWIDTH_BPS`
   - `FIXED_LATENCY_SEC`
   - `NUM_CPU_LAYERS` (for `main2.py` and `main3.py`)
   - `MID_GPU_LAYER` (for `main4.py`)
3. Observe the printed latency simulation logs and total inference time.

## Notes

- All scripts currently assume the same local model path: `/home/dingcong/models/google/gemma-2-2b-it`.
- `main4.py` simulates a multi-stage pipeline on a single physical GPU device.
- This README is intended as a quick reference for choosing and understanding each variant.
