"""Forward-pass helpers for the pipeline stages.

KV cache is the default path. Prefill calls these helpers with
past_key_values=None and a full prompt sequence. Decode calls them with the
rank-local past_key_values and a one-token query. Each rank keeps only the cache
for the decoder layers it owns.
"""

import inspect

import torch

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


def maybe_rotary_embeddings(model, hidden_states, position_ids):
    """Build rotary position embeddings when this Transformers version expects them."""
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        return None
    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        return None


def make_attention_mask(batch_size, query_len, key_value_len, dtype, device, attention_mask_2d=None):
    """Create a causal mask that supports both prefill and KV-cache decode.

    Prefill has query_len == key_value_len. Decode usually has query_len == 1
    while key_value_len is the full context length after appending the new token.
    Padding keys from left-padded batches are masked with the full 2D attention
    mask received from Rank 0.
    """
    min_value = torch.finfo(dtype).min
    query_positions = torch.arange(
        key_value_len - query_len,
        key_value_len,
        dtype=torch.long,
        device=device,
    )
    key_positions = torch.arange(key_value_len, dtype=torch.long, device=device)
    causal = key_positions.view(1, key_value_len) > query_positions.view(query_len, 1)
    mask = torch.zeros((query_len, key_value_len), dtype=dtype, device=device)
    mask = mask.masked_fill(causal, min_value)
    mask = mask.view(1, 1, query_len, key_value_len).expand(
        batch_size, 1, query_len, key_value_len
    )

    if attention_mask_2d is not None:
        padding_mask = attention_mask_2d.to(device=device)
        padding_mask = padding_mask.view(batch_size, 1, 1, key_value_len)
        mask = mask.masked_fill(padding_mask == 0, min_value)

    return mask.contiguous()


def make_position_ids(query_len, device, attention_mask_2d=None):
    """Build position ids for full prefill or one-token decode."""
    if attention_mask_2d is None:
        return torch.arange(query_len, device=device, dtype=torch.long).unsqueeze(0)

    attention_mask_2d = attention_mask_2d.to(device=device, dtype=torch.long).contiguous()
    position_ids = attention_mask_2d.cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(attention_mask_2d == 0, 0)
    return position_ids[:, -query_len:].contiguous()


def cache_position_for(query_len, key_value_len, device):
    """Return cache positions for the current query window."""
    return torch.arange(key_value_len - query_len, key_value_len, dtype=torch.long, device=device)


def get_layer_past(past_key_values, index):
    """Return the cache entry for one local layer."""
    if past_key_values is None:
        return None
    if is_transformers_cache(past_key_values):
        return past_key_values
    return past_key_values[index]


def is_transformers_cache(past_key_values):
    """Return True for newer Transformers Cache/DynamicCache objects."""
    return hasattr(past_key_values, "update") and hasattr(past_key_values, "get_seq_length")


def supports_dynamic_cache(model):
    """Detect decoder layers that use the newer cache_position-based cache API."""
    if DynamicCache is None or len(model.model.layers) == 0:
        return False
    signature = inspect.signature(model.model.layers[0].forward)
    return "cache_position" in signature.parameters


def initialize_past_key_values(model, past_key_values):
    """Create a cache object for newer Transformers versions when needed."""
    if past_key_values is not None:
        return past_key_values
    if supports_dynamic_cache(model):
        return DynamicCache()
    return None


def extract_present_key_value(layer_outputs):
    """Extract present_key_value from common Llama decoder-layer return formats."""
    if not isinstance(layer_outputs, tuple) or len(layer_outputs) < 2:
        return None
    return layer_outputs[-1]


def run_decoder_layers(
    model,
    hidden_states,
    position_ids,
    attention_mask,
    past_key_values=None,
    cache_position_len=None,
):
    """Run this rank's decoder layers and update rank-local KV cache."""
    query_len = hidden_states.shape[1]
    key_value_len = cache_position_len if cache_position_len is not None else attention_mask.shape[-1]
    position_embeddings = maybe_rotary_embeddings(model, hidden_states, position_ids)
    cache_position = cache_position_for(query_len, key_value_len, hidden_states.device)
    past_key_values = initialize_past_key_values(model, past_key_values)
    using_transformers_cache = is_transformers_cache(past_key_values)
    new_past_key_values = []

    for layer_index, layer in enumerate(model.model.layers):
        signature = inspect.signature(layer.forward)
        kwargs = {}
        if "attention_mask" in signature.parameters:
            kwargs["attention_mask"] = attention_mask
        if "position_ids" in signature.parameters:
            kwargs["position_ids"] = position_ids
        if "past_key_value" in signature.parameters:
            kwargs["past_key_value"] = get_layer_past(past_key_values, layer_index)
        if "output_attentions" in signature.parameters:
            kwargs["output_attentions"] = False
        if "use_cache" in signature.parameters:
            kwargs["use_cache"] = True
        if "cache_position" in signature.parameters:
            kwargs["cache_position"] = cache_position
        if "position_embeddings" in signature.parameters and position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings

        layer_outputs = layer(hidden_states, **kwargs)
        hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        if not using_transformers_cache:
            present_key_value = extract_present_key_value(layer_outputs)
            if present_key_value is not None:
                new_past_key_values.append(present_key_value)

    if using_transformers_cache:
        new_past_key_values = past_key_values
    elif not new_past_key_values:
        new_past_key_values = past_key_values
    else:
        new_past_key_values = tuple(new_past_key_values)
    return hidden_states, new_past_key_values


def rank0_forward(model, input_ids, device, attention_mask_2d=None, past_key_values=None):
    """Run Rank 0's embedding and early decoder layers with KV cache.

    Use the Transformers model-level forward instead of calling decoder layers
    one by one. Some Transformers versions update DynamicCache inside
    LlamaModel.forward but do not return per-layer cache objects from an
    individual LlamaDecoderLayer call. Model-level forward keeps cache, mask, and
    cache_position handling consistent with the installed library.
    """
    batch_size, query_len = input_ids.shape
    position_ids = make_position_ids(query_len, device, attention_mask_2d)
    outputs = model_model_forward(
        model,
        input_ids=input_ids,
        inputs_embeds=None,
        attention_mask_2d=attention_mask_2d,
        position_ids=position_ids,
        past_key_values=past_key_values,
    )
    return outputs[0].contiguous(), outputs[1]


def model_model_forward(
    model,
    input_ids,
    inputs_embeds,
    attention_mask_2d,
    position_ids,
    past_key_values,
):
    """Call model.model.forward with only arguments supported by this version."""
    signature = inspect.signature(model.model.forward)
    kwargs = {}
    if input_ids is not None and "input_ids" in signature.parameters:
        kwargs["input_ids"] = input_ids.contiguous()
    if inputs_embeds is not None and "inputs_embeds" in signature.parameters:
        kwargs["inputs_embeds"] = inputs_embeds.contiguous()
    if "attention_mask" in signature.parameters:
        kwargs["attention_mask"] = (
            None if attention_mask_2d is None else attention_mask_2d.contiguous()
        )
    if "position_ids" in signature.parameters:
        kwargs["position_ids"] = position_ids.contiguous()
    if "past_key_values" in signature.parameters:
        kwargs["past_key_values"] = past_key_values
    if "use_cache" in signature.parameters:
        kwargs["use_cache"] = True
    if "return_dict" in signature.parameters:
        kwargs["return_dict"] = True

    outputs = model.model(**kwargs)
    if isinstance(outputs, tuple):
        hidden_states = outputs[0]
        new_past_key_values = outputs[1] if len(outputs) > 1 else past_key_values
    else:
        hidden_states = outputs.last_hidden_state
        new_past_key_values = getattr(outputs, "past_key_values", past_key_values)
    return hidden_states, new_past_key_values


def rank_middle_forward(model, hidden_states, device, attention_mask_2d=None, past_key_values=None):
    """Run a middle pipeline rank with KV cache."""
    batch_size, query_len, _ = hidden_states.shape
    position_ids = make_position_ids(query_len, device, attention_mask_2d)
    outputs = model_model_forward(
        model,
        input_ids=None,
        inputs_embeds=hidden_states,
        attention_mask_2d=attention_mask_2d,
        position_ids=position_ids,
        past_key_values=past_key_values,
    )
    return outputs[0].contiguous(), outputs[1]


def rank1_forward_logits(model, hidden_states, device, attention_mask_2d=None, past_key_values=None):
    """Run the last pipeline rank and return last-token logits with KV cache."""
    batch_size, query_len, _ = hidden_states.shape
    position_ids = make_position_ids(query_len, device, attention_mask_2d)
    outputs = model_model_forward(
        model,
        input_ids=None,
        inputs_embeds=hidden_states,
        attention_mask_2d=attention_mask_2d,
        position_ids=position_ids,
        past_key_values=past_key_values,
    )
    hidden_states, past_key_values = outputs
    logits = model.lm_head(hidden_states)
    return logits[:, -1, :], past_key_values


def choose_next_token(logits, temperature):
    """Convert last-token logits into one token id."""
    if temperature and temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)
