import gc
import torch
from tqdm import tqdm

from utils import save_KV_to_binary_new, get_dataloader, get_few_shot
from config import *

dataloader = get_dataloader(eva_task, subsets[subset_no_to_use], train_name)

fs = 0

count = 0
ocount = 0
fcount = 0
keys = []
values = []
cache_name = model_name
cache_task = eva_task
cache_prefix = "{}/{}/".format(cache_name, cache_task)
KV_path = cache_prefix + eva_task + "Cache{}_{}s/".format("_cyc" if use_cycle else "", fs)

p = get_few_shot(fs, eva_task, subsets[subset_no_to_use])
length = len(dataloader)
bar = tqdm(total=max_generation_amount, desc="Creating Keys and Values")
for index, data in enumerate(dataloader):

    if(index < progress):
        continue

    if fcount >= max_generation_amount:
        break
    
    if use_cycle:
        option_num = len(get_data_lines(data)[1])

        if use_selected:
            prompt, cycans = p.prepare_prompt(data, False, True, 0, ufs=fs)
            ans, human = get_ans(prompt, False)
            if human or ans != cycans:
                continue
        
        for i in range(option_num):
            prompt, cycans = p.prepare_prompt(data, False, True, i, ufs=fs)

            att = get_att(prompt)
            if(len(att) == 0): continue
        
            keys.append(att)
            values.append(cycans)

            count += 1
            bar.update()

    else:
        prompt, cycans = p.prepare_prompt(data, False, ufs=fs)
        
        if use_selected:
            ans, human = get_ans(prompt, False)
            if human or ans != cycans:
                continue

        att = get_att(prompt)
        if(len(att) == 0): continue
        
        keys.append(att)
        values.append(cycans)

        count += 1
        bar.update()

    if(count < save_gap and index + 1 != length):
        continue

    save_KV_to_binary_new(keys[-save_gap:], values[-save_gap:], ocount + exists, KV_path)
    keys = keys[:-save_gap]
    values = values[:-save_gap]
    count -= save_gap
    ocount += 1
    fcount += save_gap + min(count, 0)
    gc.collect()

bar.close()
print("Done.")

print(fcount)