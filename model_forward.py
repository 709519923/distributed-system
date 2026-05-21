"""Forward-pass helpers for the pipeline stages.

This module is the right place to inspect when output quality, padding behavior,
batch inference, attention masks, position ids, or logits selection look wrong.
It contains no file I/O and no NCCL send/recv calls.
"""

import inspect

import torch


def maybe_rotary_embeddings(model, hidden_states, position_ids):
    """Build rotary position embeddings when the installed Transformers needs them.

    Different Transformers versions expose Llama/TinyLlama decoder-layer forward
    signatures slightly differently. Newer versions may pass precomputed rotary
    embeddings through position_embeddings; older versions only use position_ids.
    Returning None is fine for the older path.
    """
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        return None
    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        return None


def make_causal_mask(batch_size, seq_len, dtype, device, attention_mask_2d=None):
    """Create a causal attention mask, optionally blocking padding tokens.

    Shape is [batch, heads, query_length, key_length]. Values above the diagonal
    are set to a very negative number so a token cannot attend to future tokens.
    When attention_mask_2d is passed, key positions with value 0 are also masked.
    """
    min_value = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_value, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)

    if attention_mask_2d is not None:
        padding_mask = attention_mask_2d.to(device=device)
        padding_mask = padding_mask.view(batch_size, 1, 1, seq_len)
        mask = mask.masked_fill(padding_mask == 0, min_value)

    return mask


def make_position_ids(seq_len, device, attention_mask_2d=None):
    """Build position ids that work for both single prompts and left-padded batches."""
    if attention_mask_2d is None:
        return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    position_ids = attention_mask_2d.to(device=device, dtype=torch.long).cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask_2d.to(device=device) == 0, 0)


def run_decoder_layers(model, hidden_states, position_ids, attention_mask):
    """Run whichever decoder layers remain in this rank's model object.

    The same helper is used by both ranks. Rank 0's model contains only early
    layers; Rank 1's model contains only later layers. The inspect.signature()
    logic makes this script tolerate small API differences across Transformers
    versions without changing the core distributed logic.
    """
    position_embeddings = maybe_rotary_embeddings(model, hidden_states, position_ids)
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)

    for layer in model.model.layers:
        signature = inspect.signature(layer.forward)
        kwargs = {}
        if "attention_mask" in signature.parameters:
            kwargs["attention_mask"] = attention_mask
        if "position_ids" in signature.parameters:
            kwargs["position_ids"] = position_ids
        if "past_key_value" in signature.parameters:
            kwargs["past_key_value"] = None
        if "output_attentions" in signature.parameters:
            kwargs["output_attentions"] = False
        if "use_cache" in signature.parameters:
            kwargs["use_cache"] = False
        if "cache_position" in signature.parameters:
            kwargs["cache_position"] = cache_position
        if "position_embeddings" in signature.parameters and position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings

        layer_outputs = layer(hidden_states, **kwargs)
        hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

    return hidden_states


def rank0_forward(model, input_ids, device, attention_mask_2d=None):
    """Run Rank 0's part of the model and return hidden states for Rank 1.

    Rank 0 starts from token ids, so it must apply token embedding first. It then
    runs the early decoder layers and sends the resulting hidden_states tensor to
    Rank 1 through NCCL.
    """
    batch_size, seq_len = input_ids.shape
    position_ids = make_position_ids(seq_len, device, attention_mask_2d)
    attention_mask = make_causal_mask(batch_size, seq_len, model.dtype, device, attention_mask_2d)

    hidden_states = model.model.embed_tokens(input_ids)
    hidden_states = run_decoder_layers(model, hidden_states, position_ids, attention_mask)
    return hidden_states.contiguous()


def rank_middle_forward(model, hidden_states, device, attention_mask_2d=None):
    """Run a middle pipeline rank: hidden_states -> local decoder layers."""
    batch_size, seq_len, _ = hidden_states.shape
    position_ids = make_position_ids(seq_len, device, attention_mask_2d)
    attention_mask = make_causal_mask(batch_size, seq_len, model.dtype, device, attention_mask_2d)

    hidden_states = run_decoder_layers(model, hidden_states, position_ids, attention_mask)
    return hidden_states.contiguous()


def rank1_forward_logits(model, hidden_states, device, attention_mask_2d=None):
    """Run the last pipeline rank and return logits for the last token.

    The last rank receives hidden states, not token ids. Therefore it skips embeddings,
    runs the later decoder layers, applies final norm and lm_head, then returns
    only the last-token logits needed to choose the next generated token.
    """
    batch_size, seq_len, _ = hidden_states.shape
    position_ids = make_position_ids(seq_len, device, attention_mask_2d)
    attention_mask = make_causal_mask(batch_size, seq_len, model.dtype, device, attention_mask_2d)

    hidden_states = run_decoder_layers(model, hidden_states, position_ids, attention_mask)
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits[:, -1, :]


def choose_next_token(logits, temperature):
    """Convert last-token logits into one token id.

    temperature=0 uses greedy decoding. A positive temperature samples from the
    softmax distribution, which makes output less deterministic.
    """
    if temperature and temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)
