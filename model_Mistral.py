import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

dimension = 4096                            # Attention dimension
attention_heads = 32                        # How many attention heads to use
head_size = dimension // attention_heads    # How many dimension in each attention head
max_length = 4096
model_name = "mistral"
tempreture = 160

device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained("./mistral/Mistral-7B-v0.1", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./mistral/Mistral-7B-v0.1", trust_remote_code=True, low_cpu_mem_usage=True, device_map="auto").half()
model.eval()


@torch.no_grad()
def get_att(prompt: str) -> list:
    
    # ips = torch.cat([tokenizer(prompt, return_tensors="pt")["input_ids"], torch.LongTensor([[187]])], 
    #                 dim=-1)
    # prompt = cut_prompt(prompt)
    ips = tokenizer(prompt, return_tensors="pt")["input_ids"]
    base_length = ips.shape[-1]

    if(base_length > max_length):
        return []
    
    att = model.model(
        input_ids=ips.to(device),
    ).last_hidden_state[0][-1].cpu().tolist()

    return att

choices = {68 : 0, 69 : 1, 70 : 2, 71 : 3, 72 : 4, 73 : 5, 74 : 6, 
           330: 0, 365: 1, 334: 2, 384: 3, 413: 4, 401: 5, 420: 6, 
           28741: 0, 28760: 1, 28743: 2, 28757: 3, 28749: 4, 28765: 5, 28777: 6
           }

@torch.no_grad()
def get_length(prompt: str) -> int:
    # prompt = cut_prompt(prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs["input_ids"].shape[-1]

logits_temp = 100
e = 2.7181718
@torch.no_grad()
def get_ans(prompt: str, full_generate=True):

    global logitss
    
    # This model dose not support full_genetare.

    # prompt = cut_prompt(prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    base_length = inputs["input_ids"].shape[-1]

    if base_length > max_length:
        return 0, -1, 0

    # if full_generate:
    #     option = model.generate(inputs.input_ids.to(device), 
    #                             return_dict=False, 
    #                             max_length=4096)[0][base_length:].cpu().tolist()[0]
    # else:
    #     option = model(inputs.input_ids.to(device), 
    #                    return_dict=False)[0][0][-1].cpu().argmax().item()
    
    logits = model(inputs["input_ids"].to(device))["logits"].cpu()[0][-1]
    option = logits.argmax(dim=-1).item()

    logits = logits / logits_temp
    cppl = (e ** logits).sum().item()

    human = 0
    answer = choices.get(option)
    if answer is None:
        human = 1
        answer = -1
    
    return answer, human, cppl

if __name__ == "__main__":

    text = '''What's the right anser? Output A B C or D.
A. The wrong answer.
B. The right answer.
C. The wrong answer.
D. The wrong answer.
Answer:
'''
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    print(input_ids)
    output = model(input_ids.to(device))["logits"][0].cpu()
    o_tokens = output.argmax(dim=-1)
    print(o_tokens.shape)
    print(tokenizer.decode(o_tokens))

    print(get_att(text), len(get_att(text)))
    print(get_ans(text))

    # for k, v in choices.items():
    #     print("Key: {} Val: {} Chr: {}".format(k, v, tokenizer.decode(k)))

    # for i in range(50000):
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

    