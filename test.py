import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained("./mistral/Mistral-7B-v0.1", trust_remote_code=True)

print(tokenizer("Harmony", return_tensors="pt")["input_ids"])
print(tokenizer("oy", return_tensors="pt")["input_ids"])
print(tokenizer.decode([26988]), tokenizer.decode([2557]))