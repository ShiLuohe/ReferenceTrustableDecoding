import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig

import os
import torch
import platform
from colorama import Fore, Style

dimension = 4096                            # Attention dimension
attention_heads = 32                        # How many attention heads to use
head_size = dimension // attention_heads    # How many dimension in each attention head
max_length = 4096
model_name = "baichuan2"
tempreture = 1000

device = torch.device('cuda:1')
tokenizer = AutoTokenizer.from_pretrained("baichuan2/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan2/Baichuan2-7B-Chat", 
                                             torch_dtype=torch.bfloat16, 
                                             trust_remote_code=True
                                             ).half().to(device)
model.generation_config = GenerationConfig.from_pretrained("baichuan2/Baichuan2-7B-Chat")
model.eval()

@torch.no_grad()
def get_att(prompt: str) -> list:
    
    ips = torch.cat([tokenizer(prompt, return_tensors="pt")["input_ids"], torch.LongTensor([[5]])], 
                    dim=-1)

    if(ips.shape[-1] > max_length):
        return []
    
    att = model.model(
        ips.to(device),
    ).last_hidden_state[0][-1].cpu().tolist()

    return att

@torch.no_grad()
def get_length(prompt: str) -> int:
    inputs = tokenizer(prompt, return_tensors="pt")
    return  inputs["input_ids"].shape[-1]

choices = {1154: 0, 1155: 1, 1156: 2, 1157: 3, 1158: 4, 1159: 5, 1160: 6, 
           1399: 2, 1401: 0, 1432: 1, 1443: 3, 1454: 5, 1485: 6, 1486: 4, 
           92343: 0, 92344: 2, 92359: 1, 92363: 3, 92367: 4, 92379: 5, 92394: 6,}
@torch.no_grad()
def get_ans(prompt: str, full_generate=True):

    ips = torch.cat([tokenizer(prompt, return_tensors="pt")["input_ids"], torch.LongTensor([[5]])], 
                    dim=-1)
    
    if(ips.shape[-1] > max_length):
        return 0, -1

    option = 0

    # if full_generate:
    #     option = model.chat(tokenizer, prompt, history=[])[0][0]
    # else:
    # output = model.transformer(inputs["input_ids"].to(device), 
    #                 return_dict=False)[0][-1][0]
    option = model(ips.to(device))["logits"].cpu().argmax(dim=-1)[0][-1].item()
    
    human = 0
    answer = choices.get(option)
    if answer is None:
        human = 1
        answer = -1
    
    return answer, human


if __name__ == "__main__":

    text = '''What's the right answer? Output A B C or D.
A. The wrong answer.
B. The right answer.
C. The wrong answer.
D. The wrong answer.
Answer:
'''
    input_ids = torch.cat([tokenizer(text, return_tensors="pt")["input_ids"],
                           torch.LongTensor([[5]])],
                          dim=-1
                          )
    print(input_ids)
    output = model(input_ids.to(device))["logits"][0].cpu()
    o_tokens = output.argmax(dim=-1)
    print(o_tokens.shape)
    print(tokenizer.decode(o_tokens))

    print(get_att(text), len(get_att(text)))
    print(get_ans(text))

    for k, v in choices.items():
        print("Key: {} Val: {} Chr: {}".format(k, v, tokenizer.decode(k)))

    # for i in range(125000):
    #     token = tokenizer.decode([i])
    #     if token == 'A' or token == ' A' or token == 'A.':
    #         print(f"{i}: 0, ", end='')
    #     if token == 'B' or token == ' B':
    #         print(f"{i}: 1, ", end='')
    #     if token == 'C' or token == ' C':
    #         print(f"{i}: 2, ", end='')
    #     if token == 'D' or token == ' D':
    #         print(f"{i}: 3, ", end='')
    #     if token == 'E' or token == ' E':
    #         print(f"{i}: 4, ", end='')
    #     if token == 'F' or token == ' F':
    #         print(f"{i}: 5, ", end='')
    #     if token == 'G' or token == ' G':
    #         print(f"{i}: 6, ", end='')
    # print()
