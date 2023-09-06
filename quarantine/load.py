import torch
from torch import nn
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig


# Takes a path of the model
# Returns a state dict of the model
def load_hf_llama_state_dict(model_path):
    # TODO: Handle bnb at seperate layer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = LlamaForCausalLM.from_pretrained(model_path, device_map={"": 0},  torch_dtype=torch.float16)
    return model.state_dict()

class SterilizedHFLlamaTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token_id
        self.pad_token = tokenizer.pad_token_id

    def encode(self, text, return_tensors="pt", padding=True, device=0):
        output = self.tokenizer(text, return_tensors=return_tensors, padding=padding).to(device)
        return output['input_ids'], output['attention_mask']

        
    def decode(self, outputs, skip_special_tokens=True):
        return self.tokenizer.decode(outputs, skip_special_tokens=skip_special_tokens)

# Takes a path of the tokenizer model
# Returns an object, with two functions, decode and encode
# Decode returns a dictionary of things. I trust that my child assigns them to adequately named variables..
def load_sterilized_hf_llama_tokenizer(model_path):
    tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token
    return SterilizedHFLlamaTokenizer(tokenizer)

# Usage
# model_path = "/home/kache/models/llama1/7bhf/model"
# tokenizer_path = "/home/kache/models/llama1/7bhf/tokenizer"
# sterilized_tokenizer = load_sterilized_hf_llama_tokenizer(tokenizer_path)
# model = load_hf_llama_state_dict(model_path)