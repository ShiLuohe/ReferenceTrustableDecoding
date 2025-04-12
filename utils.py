import struct
from tqdm import tqdm
import gc
import torch
import numpy as np

from config import *

file_name = "{}_{}-{}.{}"

def save_KV_to_binary(keys, values, start, size, path):

    keys_path = path + file_name.format("keys", start, start + size - 1, "bin")
    values_path = path + file_name.format("values", start, start + size - 1, "bin")
    config_path = path + file_name.format("config", start, start + size - 1, "txt")

    total = len(values)

    with open(values_path, 'wb') as values_file:
        for int_number in tqdm(values, total=total, desc="Writing values"):
            binary_data = struct.pack('i', int_number)
            values_file.write(binary_data)
    
    with open(keys_path, 'wb') as keys_file:
        for float_numbers in tqdm(keys, total=total, desc="Writing keys  "):
            for float_number in float_numbers:
                binary_data = struct.pack('f', float_number)
                keys_file.write(binary_data)
    
    with open(config_path, 'w') as config_file:
        print(total, file=config_file)

    return

def load_KV_from_binary(start, size, path, of_batch: int, batch_count: int):

    float_size = struct.calcsize('f')
    int_size = struct.calcsize('i')

    keys_path = path + file_name.format("keys", start, start + size - 1, "bin")
    values_path = path + file_name.format("values", start, start + size - 1, "bin")
    config_path = path + file_name.format("config", start, start + size - 1, "txt")

    keys = []
    values = []

    config_file = open(config_path, 'r')
    total = int(config_file.readline())

    with open(values_path, 'rb') as values_file:
        binary_data = values_file.read()
        for i in tqdm(range(total), 
                      total=total, 
                      desc="Reading values ({}/{})".format(of_batch, batch_count), 
                      leave=False
                      ):
            start = i * int_size
            end = start + int_size
            int_bytes = binary_data[start:end]
            int_value = struct.unpack('i', int_bytes)[0]
            values.append(int_value)
    
    with open(keys_path, 'rb') as keys_file:
        binary_data = keys_file.read()
        # print(len(binary_data))
        for j in tqdm(range(total), 
                      total=total, 
                      desc="Reading keys   ({}/{})".format(of_batch, batch_count), 
                      leave=False
                      ):
            key = []
            offset = j * 4096 * float_size
            for i in range(dimension):
                start = i * float_size + offset
                end = start + float_size
                float_bytes = binary_data[start:end]
                float_value = struct.unpack('f', float_bytes)[0]
                key.append(float_value)
            keys.append(key)

            if j % 1024 == 0:
                gc.collect()

    return keys, values

file_name_new = "{}_{}.{}"

def save_KV_to_binary_new(keys, values, num, path):

    keys_path = path + file_name_new.format("keys", num, "bin")
    values_path = path + file_name_new.format("values", num, "bin")
    config_path = path + file_name_new.format("config", num, "txt")

    total = len(values)

    with open(values_path, 'wb') as values_file:
        for int_number in tqdm(values, total=total, desc="Writing values"):
            binary_data = struct.pack('i', int_number)
            values_file.write(binary_data)
    
    with open(keys_path, 'wb') as keys_file:
        for float_numbers in tqdm(keys, total=total, desc="Writing keys  "):
            for float_number in float_numbers:
                binary_data = struct.pack('f', float_number)
                keys_file.write(binary_data)
    
    with open(config_path, 'w') as config_file:
        print(total, file=config_file)

    return

def load_KV_from_binary_new(num, path, of_batch: int, batch_count: int):

    float_size = struct.calcsize('f')
    int_size = struct.calcsize('i')

    keys_path = path + file_name_new.format("keys", num, "bin")
    values_path = path + file_name_new.format("values", num, "bin")
    config_path = path + file_name_new.format("config", num, "txt")

    keys = []
    values = []

    config_file = open(config_path, 'r')
    total = int(config_file.readline())

    with open(values_path, 'rb') as values_file:
        binary_data = values_file.read()
        for i in tqdm(range(total), 
                      total=total, 
                      desc="Reading values ({}/{})".format(of_batch, batch_count), 
                      leave=False
                     ):
            start = i * int_size
            end = start + int_size
            int_bytes = binary_data[start:end]
            int_value = struct.unpack('i', int_bytes)[0]
            values.append(int_value)
    
    with open(keys_path, 'rb') as keys_file:
        binary_data = keys_file.read()
        # print(len(binary_data))
        for j in tqdm(range(total), 
                      total=total, 
                      desc="Reading keys   ({}/{})".format(of_batch, batch_count), 
                      leave=False
                     ):
            key = []
            offset = j * dimension * float_size
            for i in range(dimension):
                start = i * float_size + offset
                end = start + float_size
                float_bytes = binary_data[start:end]
                float_value = struct.unpack('f', float_bytes)[0]
                key.append(float_value)
            keys.append(key)

            if (j + 1) % 1024 == 0:
                gc.collect()

    return keys, values

# =============================================================

choice_head = {0: '\nA. ', 1: '\nB. ', 2: '\nC. ', 3: '\nD. ',
               4: '\nE. ', 5: '\nF. ', 6: '\nG. ', 7: '\nH. ',
               8: '\nI. ', 9: '\nJ. ',10: '\nL. ',11: "\nM."}
class prompt_pipline:

    def __init__(self) -> None:
        self.chat_prompt = question_format
        self.base_prompts = []
        self.max_shot = 0
    
    def prepare_prompt_core(self, data, include_answer=False, cycling=0) -> tuple[str, int]:

        question_prompt, choices = get_data_lines(data)
        choices_prompt = ''
        answer = get_data_answer(data)
        option_num = len(choices)

        for index in range(option_num):
            choices_prompt += choice_head[index]
            choices_prompt += choices[(index + cycling) % option_num]

        final = ""
        if include_answer:
            final += choice_head[(answer - cycling + option_num) % option_num]
            final += choices[answer]
            final += '\n'
        
        cycans = (answer - cycling + option_num) % option_num
        return self.chat_prompt.format(question_prompt, choices_prompt, final), cycans

    def prepare_base_prompt(self, usf: int) -> str:
        ret = ""
        if usf > self.max_shot: usf = self.max_shot
        for i in range(usf):
            ret += self.base_prompts[i]
        return ret

    def prepare_prompt(self, data, include_answer=False, use_base_prompt=True, cycling=0, ufs=0):
        ques_prompt, cycans = self.prepare_prompt_core(data, include_answer, cycling)
        if use_base_prompt:
            return question_head + self.prepare_base_prompt(ufs) + ques_prompt, cycans
        else:
            return ques_prompt, cycans

    def add_shot(self, data):
        self.base_prompts.append(self.prepare_prompt_core(data, True, False)[0])
        self.max_shot += 1

    def clear_shot(self):
        self.base_prompt = []
        self.max_shot = 0

    def print_base(self):
        file = open("prompt.txt", "w")
        print(self.prepare_base_prompt(self.max_shot), file=file)
        file.close()

# =============================================================
def get_few_shot(shots: int, task: str , subset: str, use_train=False) -> prompt_pipline:
    
    ret = prompt_pipline()
    p = prompt_pipline()
    if shots == 0:
        return prompt_pipline()

    max_single_length = max_length // (shots + 1)
    dataloader = get_dataloader(task, subset, vali_name)
    if use_train:
        dataloader = get_dataloader(task, subset, train_name)
    WAdatas = []
    RTdatas = []

    total = 0
    for data in tqdm(dataloader, desc="Preparing Few Shots........", leave=False):

        prompt, canswer = ret.prepare_prompt(data)
        ans, ff = get_ans(prompt, full_generate=False)
        if ff:
            if get_length(p.prepare_prompt(data)[0]) > max_single_length: continue
            ret.add_shot(data)
            total += 1
            if total >= shots: break
        elif ans != canswer:
            WAdatas.append(data)
        else:
            RTdatas.append(data)
    
    for WAdata in WAdatas:
        if total >= shots: break
        if get_length(p.prepare_prompt(WAdata)[0]) > max_single_length: continue
        ret.add_shot(WAdata)
        total += 1
    
    for RTdata in RTdatas:
        if total >= shots: break
        if get_length(p.prepare_prompt(RTdata)[0]) > max_single_length: continue
        ret.add_shot(RTdata)
        total += 1

    return ret

# =============================================================
# Abandoned function
def examine(words: list):

    if len(words) <= 1:
        return -1, 0, 0

    offset = 0

    if words[0] == 'Multihead': offset = 0
    elif words[0] == 'Naive': offset = 1
    elif words[0] == 'Full': offset = 2
    elif words[0] == 'First': offset = 3
    elif words[1] == 'full': return 4, int(words[5]), 0
    elif words[1] == 'first': return 5, int(words[5]), 0
    else: return -1, 0, 0

    acc = int(words[4])
    total = int(words[4 + 1])

    return offset, acc, total

# Abandoned function
def ana(raw_file_path: str, output_file_path: str):

    file = open(raw_file_path, "r")
    lines = file.readlines()
    file.close()

    accs = [[], [], [], []]
    tots = [[], [], [], []]
    human1 = []
    human2 = []
    acc = [0, 0, 0, 0, 0, 0]
    tot = [0, 0, 0, 0, 0, 0]
    score = [0, 0, 0, 0, 0, 0]

    for line in lines:
        words = line.split()

        cate, tacc, ttot = examine(words)

        if cate == 4:
            human1.append(tacc)
            continue
        elif cate == 5:
            human2.append(tacc)
            continue
        elif cate == -1:
            continue
        
        acc[cate] += tacc
        accs[cate].append(tacc)

        tot[cate] += ttot
        tots[cate].append(ttot)
    
    file = open(output_file_path, "w")

    count = len(accs[0])
    for i in range(count):
        posmax1 = accs[2][i] + accs[2][i] / (tots[2][i] - human1[i]) * human1[i]
        posmax2 = accs[3][i] + accs[3][i] / (tots[3][i] - human2[i]) * human2[i]

        acc[4] += posmax1
        tot[4] += tots[2][i]
        
        acc[5] += posmax2
        tot[5] += tots[3][i]

        print(accs[0][i], accs[1][i], accs[2][i], accs[3][i], posmax1, posmax2, file=file)
        print(accs[0][i] / tots[0][i],
              accs[1][i] / tots[1][i],
              accs[2][i] / tots[2][i], 
              accs[3][i] / tots[3][i],
              posmax1 / tots[2][i],
              posmax2 / tots[3][i], file=file, end='\n\n')
        score[0] += accs[0][i] / tots[0][i]
        score[1] += accs[1][i] / tots[1][i]
        score[2] += accs[2][i] / tots[2][i]
        score[3] += accs[3][i] / tots[3][i]
        score[4] += posmax1 / tots[2][i]
        score[5] += posmax2 / tots[3][i]

    print("================", file=file)
    rate = [tacc/ttot for tacc, ttot in zip(acc, tot)]
    print(acc, file=file)
    print(tot, file=file)
    print(rate, file=file)
    scores = [pt / count for pt in score]
    print(scores, file=file)

# =============================================================
def getbest():
    prof = []
    for i in range(32):
        str = input().split()
        prof.append(float(str[2]))
    ts = torch.tensor(prof)
    for i in range(32):
        maxi = ts.argmax(dim=-1)
        print(maxi.item(), end=' ')
        ts[maxi] = 0


if __name__ == "__main__":
    p = get_few_shot(5, eva_task, "ARC-Challenge")
    dataloader = get_dataloader(eva_task, "ARC-Challenge", "train")
    for data in dataloader:
        for i in range(2):
            print(p.prepare_prompt(data, include_answer=True, use_base_prompt=True,cycling=i)[0])
            input("=====")
    # view_dataset()

'''

[5620, 5504, 4261, 5783.154363501783]
[14033, 14033, 14042, 14042]
[0.4004845720800969, 0.39221834247844367, 0.3034468024497935, 0.4118469137944583]

'''
    


