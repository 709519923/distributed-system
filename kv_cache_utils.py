"""Small measurement helpers for KV-cache experiments.

These helpers avoid touching distributed control flow. If a reported parameter
size, KV-cache size, tensor size, or CUDA memory number looks odd, start here.
All capacity values are first measured in bytes and later rendered as MB.
"""

import torch


def tensor_bytes(tensor):
    """Return the allocated byte size for one tensor-like object."""
    if not torch.is_tensor(tensor):
        return 0
    return int(tensor.numel() * tensor.element_size())


def bytes_to_mb(value):
    """Convert bytes to MiB-style MB for human-readable experiment logs."""
    return float(value) / 1024.0 / 1024.0


def count_model_parameters(model):
    """Count parameters that remain in the current rank's pruned model stage."""
    return sum(param.numel() for param in model.parameters())


def estimate_parameter_bytes(model):
    """Estimate parameter memory for the current rank's model stage."""
    return sum(tensor_bytes(param) for param in model.parameters())


def estimate_past_key_values_bytes(past_key_values):
    """Recursively estimate KV-cache memory from tuples/lists/dicts/tensors."""
    if past_key_values is None:
        return 0
    if torch.is_tensor(past_key_values):
        return tensor_bytes(past_key_values)
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return estimate_past_key_values_bytes(past_key_values.key_cache) + estimate_past_key_values_bytes(
            past_key_values.value_cache
        )
    if hasattr(past_key_values, "layers"):
        return estimate_past_key_values_bytes(past_key_values.layers)
    if hasattr(past_key_values, "keys") and hasattr(past_key_values, "values"):
        return estimate_past_key_values_bytes(past_key_values.keys) + estimate_past_key_values_bytes(
            past_key_values.values
        )
    if isinstance(past_key_values, dict):
        return sum(estimate_past_key_values_bytes(value) for value in past_key_values.values())
    if isinstance(past_key_values, (list, tuple)):
        return sum(estimate_past_key_values_bytes(value) for value in past_key_values)
    return 0


def cuda_memory_allocated(device):
    """Return current CUDA allocated memory in bytes."""
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_allocated(device))


def cuda_memory_reserved(device):
    """Return current CUDA reserved memory in bytes."""
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_reserved(device))


def synchronize_cuda():
    """Synchronize CUDA before timing or reading memory counters."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
