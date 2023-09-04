# Why? I got a little annoyed w/ huggingface's.. code. So I ripped it out and fixed some over abstraction
# The end goal is to remove all dependencies from huggingface.
# Still not completely unfucked!

# Attribution:
# Most of this code has been ripped out of huggingface, with great pain
# Shout out to kaparthy's GPT & makemore series. Helped a lot
# Shout out to llama implementation from meta
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""

import torch
import math
from torch import nn
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

model_id = "/home/kache/models/llama2/7bhf/model"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = LlamaTokenizer.from_pretrained(
    "/home/kache/models/llama2/7bhf/tokenizer", legacy=True)
model = LlamaForCausalLM.from_pretrained(
    "/home/kache/models/llama2/7bhf/model", device_map={"": 0},  quantization_config=bnb_config)
print(model)

question = [
    "I went to the store the other day",
    "For what its worth, I really dont think that you should"
]
tokenizer.pad_token = tokenizer.eos_token


def generate(
    input_ids = None,
    max_length = None,
    attention_mask = None,
    use_cache = None,
    model_as_fn = None,
    model = None,
):
    generation_config = model.generation_config

    pad_token_id = generation_config.pad_token_id
    eos_token_id = generation_config.eos_token_id
    eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device)

    # We want to mark sequences that are finished
    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    past_key_values = None
    position_ids = None

    while True:
        # todo: This used to be worse, but its still bad. continue unfucking
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
                "use_cache": True,
                "attention_mask": attention_mask,
            }
        )

        outputs = model_as_fn(
            model,
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


def model_as_fn(
    model,
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

    # Word embedding
    hidden_states = model.model.embed_tokens(input_ids)

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

    for i, decoder_layer in enumerate(model.model.layers):
        past_key_value = past_key_values[i] if past_key_values is not None else None
        residual = hidden_states

        def rmsnorm(model, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * \
                torch.rsqrt(variance + model.variance_epsilon)
            return model.weight * hidden_states.to(input_dtype)

        hidden_states = rmsnorm(decoder_layer.input_layernorm, hidden_states)

        # Self Attention

        def self_attn(
            model,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
        ):
            def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
                """
                This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
                num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
                """
                batch, num_key_value_heads, slen, head_dim = hidden_states.shape
                if n_rep == 1:
                    return hidden_states
                hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
                return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

            B, q_len, _ = hidden_states.size()
            query_states = model.q_proj(hidden_states)
            key_states = model.k_proj(hidden_states)
            value_states = model.v_proj(hidden_states)

            query_states = query_states.view(
                B, q_len, model.num_heads, model.head_dim).transpose(1, 2)
            key_states = key_states.view(
                B, q_len, model.num_key_value_heads, model.head_dim).transpose(1, 2)
            value_states = value_states.view(
                B, q_len, model.num_key_value_heads, model.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]

            cos, sin = model.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids)

            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat(
                    [past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, model.num_key_value_groups)
            value_states = repeat_kv(value_states, model.num_key_value_groups)
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)) / math.sqrt(model.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(B, q_len, model.hidden_size)

            attn_output = model.o_proj(attn_output)

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

        # Fully Connected
        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)

        def mlp(model, x):
            down_proj = model.down_proj(model.act_fn(
                model.gate_proj(x)) * model.up_proj(x))
            return down_proj

        hidden_states = mlp(decoder_layer.mlp, hidden_states)
        hidden_states = residual + hidden_states
        layer_outputs = (hidden_states,)

        if use_cache:
            layer_outputs += (present_key_value,)

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (
                layer_outputs[1],)

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    logits = logits.float()

    return CausalLMOutputWithPast(
        logits=logits,
        past_key_values=next_decoder_cache,
    )


inputs = tokenizer(question, return_tensors="pt", padding=True).to(0)
outputs = generate(
    input_ids = inputs.input_ids,
    max_length = 24,
    attention_mask = inputs.attention_mask,
    use_cache = True,
    model_as_fn = model_as_fn,
    model = model,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(tokenizer.decode(outputs[1], skip_special_tokens=True))
