# Why? I got a little annoyed w/ huggingface's.. code. So I ripped it out and fixed some over abstraction
# The end goal is to remove all dependencies from huggingface.

# Attribution:
# Most of this code has been ripped out of huggingface, with great pain
# Shout out to kaparthy's GPT & makemore series. Helped a lot
# Shout out to llama implementation from meta
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.

import torch
import math
from box import Box # Forgive me
from torch import nn
from torch.nn.functional import silu
from transformers.modeling_outputs import CausalLMOutputWithPast
from quarantine.load import load_hf_llama_state_dict, load_sterilized_hf_llama_tokenizer

def generate(
    input_ids = None,
    max_length = None,
    attention_mask = None,
    use_cache = None,
    model_as_fn = None,
    model_state_dict = None,
    pad_token_id = None,
    eos_token_id = None,
):
    eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device)

    # We want to mark sequences that are finished
    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    past_key_values = None
    position_ids = None

    while True:
        # todo: This used to be worse, but its still bad. continue healing
        if past_key_values:
            inference_state = {"input_ids": input_ids[:, -1:]}
        else:
            inference_state = {"input_ids": input_ids}

        if attention_mask is not None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        inference_state.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )

        outputs = model_as_fn(
            model_state_dict,
            input_ids = inference_state['input_ids'],
            position_ids = inference_state['position_ids'],
            past_key_values = inference_state['past_key_values'],
            use_cache = inference_state['use_cache'],
            attention_mask = inference_state['attention_mask'],
        )

        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + \
            pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        past_key_values = outputs.past_key_values

        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
        )

        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(
                eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )

        # stop when each sentence is finished
        if unfinished_sequences.max() == 0:
            break

        # # stop if we exceed the maximum length
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= max_length
        if is_done:
            break

    return input_ids

def get_total_layers(model_state_dict):
    layer_keys = [key for key in model_state_dict.keys() if "model.layers." in key]
    layer_numbers = set([int(key.split('.')[2]) for key in layer_keys])
    total_layers = max(layer_numbers) + 1
    return total_layers

def get_layer_weights(layer_number, model_state_dict):
    layer_key = f"model.layers.{layer_number}"
    weights = {
        "self_attn": {
            "q_proj": model_state_dict[f"{layer_key}.self_attn.q_proj.weight"],
            "k_proj": model_state_dict[f"{layer_key}.self_attn.k_proj.weight"],
            "v_proj": model_state_dict[f"{layer_key}.self_attn.v_proj.weight"],
            "o_proj": model_state_dict[f"{layer_key}.self_attn.o_proj.weight"],
        },
        "mlp": {
            "gate_proj": model_state_dict[f"{layer_key}.mlp.gate_proj.weight"],
            "up_proj": model_state_dict[f"{layer_key}.mlp.up_proj.weight"],
            "down_proj": model_state_dict[f"{layer_key}.mlp.down_proj.weight"],
        },
        "input_layernorm": model_state_dict[f"{layer_key}.input_layernorm.weight"],
        "post_attention_layernorm": model_state_dict[f"{layer_key}.post_attention_layernorm.weight"]
    }

    ret = Box(weights)
    return ret


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

def rmsnorm(weights, hidden_states, rms_norm_epsilon=1e-05):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + rms_norm_epsilon)
    return weights * hidden_states.to(input_dtype)

def model_as_fn(
    model_state_dict,
    input_ids,
    position_ids,
    past_key_values,
    use_cache,
    attention_mask,
):
    B, T = input_ids.shape
    sequence_length_with_past = T
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values else 0
    sequence_length_with_past += past_key_values_length if past_key_values else 0

    embedding_layer = nn.Embedding.from_pretrained(model_state_dict["model.embed_tokens.weight"])
    hidden_states = embedding_layer(input_ids)

    # Attention mask
    def prepare_attention_mask(attention_mask, input_shape, input_embeds, past_key_values_length):
        B, source_length = attention_mask.size()
        target_length = input_shape[-1]
        dtype = input_embeds.dtype
        device = input_embeds.device
        mask = torch.full((target_length, target_length),
                          torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (
            mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(
                target_length, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        combined_attention_mask = mask[None, None, :, :].expand(
            B, 1, target_length, target_length + past_key_values_length)
        expanded_mask = attention_mask[:, None, None, :].expand(
            B, 1, target_length, source_length).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        expanded_attn_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
            combined_attention_mask
        )
        return combined_attention_mask

    attention_mask = prepare_attention_mask(
        attention_mask, (B, T), hidden_states, past_key_values_length)

    next_decoder_cache = () if use_cache else None

    total_layers = get_total_layers(model_state_dict)
    for i in range(total_layers):
        decoder_layer = get_layer_weights(i, model_state_dict)
        past_key_value = past_key_values[i] if past_key_values is not None else None
        residual = hidden_states
        hidden_states = rmsnorm(decoder_layer.input_layernorm, hidden_states)

        # Self Attention
        def self_attn(
            layer_weights,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
            num_key_value_heads=32,
            num_attention_heads=32,
            hidden_size=4096,
            max_position_embeddings=4096,
            rope_theta=10000
        ):
            head_dim = hidden_size // num_attention_heads
            def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
                """
                This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
                num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
                # so why is it differetnt?
                # TODO: understand and avoid ugly
                """
                batch, num_key_value_heads, slen, head_dim = hidden_states.shape
                if n_rep == 1:
                    return hidden_states
                hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
                return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

            B, q_len, _ = hidden_states.size()
            query_states = hidden_states @ layer_weights.q_proj.T
            key_states = hidden_states @ layer_weights.k_proj.T
            value_states = hidden_states @ layer_weights.v_proj.T

            query_states = query_states.view(
                B, q_len, num_attention_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(
                B, q_len, num_key_value_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(
                B, q_len, num_key_value_heads, head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]

            def get_rotary_embedding(value_states, dim, seq_len, max_position_embeddings, device, base=10000):
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
                t = torch.arange(max_position_embeddings, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                # TODO: Avoid recomputing somehow
                # How much faster is it *really*?
                cos = emb.cos()[None, None, :, :].to(inv_freq.dtype)
                sin = emb.sin()[None, None, :, :].to(inv_freq.dtype)
                return cos[:, :, :seq_len, ...].to(dtype=value_states.dtype), sin[:, :, :seq_len, ...].to(dtype=value_states.dtype)

            cos, sin = get_rotary_embedding(
                value_states,
                head_dim,
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
                device=input_ids.device,
                seq_len=kv_seq_len
            )

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids)

            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat(
                    [past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            num_key_value_groups = num_attention_heads // num_key_value_heads
            # Should be nil op in the case of 7b
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            # TODO: Why? Not sure why! I cargo'd from huggingface
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(B, q_len, hidden_size)

            attn_output = attn_output @ layer_weights.o_proj.T

            return attn_output, past_key_value

        hidden_states, present_key_value = self_attn(
            decoder_layer.self_attn,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = rmsnorm(decoder_layer.post_attention_layernorm, hidden_states)

        def mlp_new(layer_weights, x):
            up_proj_output = x @ layer_weights.up_proj.T
            gate_proj_output = x @ layer_weights.gate_proj.T
            down_proj_input = silu(gate_proj_output) * up_proj_output
            down_proj_output = down_proj_input @ layer_weights.down_proj.T
            return down_proj_output

        hidden_states = mlp_new(decoder_layer.mlp, hidden_states)
        hidden_states = residual + hidden_states
        layer_outputs = (hidden_states,)

        if use_cache:
            layer_outputs += (present_key_value,)

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (
                layer_outputs[1],)

    norm_weights = model_state_dict["model.norm.weight"]
    lm_head_weights = model_state_dict["lm_head.weight"]
    hidden_states = rmsnorm(norm_weights, hidden_states)

    logits = hidden_states @ lm_head_weights.T

    return CausalLMOutputWithPast(
        logits=logits,
        past_key_values=next_decoder_cache,
    )

model_path = "/home/kache/models/llama2/7bhf/model"
tokenizer_path = "/home/kache/models/llama2/7bhf/tokenizer"
tokenizer = load_sterilized_hf_llama_tokenizer(tokenizer_path)
model_state_dict = load_hf_llama_state_dict(model_path)

prompts = [
    "I went to the store the other day",
    "For what its worth, I really dont think that you should"
]
inputs, attention_mask = tokenizer.encode(prompts)
outputs = generate(
    input_ids = inputs,
    max_length = 100,
    attention_mask = attention_mask,
    use_cache = True,
    model_as_fn = model_as_fn,
    model_state_dict = model_state_dict,
    eos_token_id=tokenizer.eos_token,
    pad_token_id=tokenizer.pad_token,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(tokenizer.decode(outputs[1], skip_special_tokens=True))
