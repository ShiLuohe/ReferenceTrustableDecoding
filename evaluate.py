from tqdm import tqdm
import torch
import numpy as np
import faiss

from utils import load_KV_from_binary_new, get_few_shot, prompt_pipline, save_KV_to_binary_new
from config import *

# Faiss library
gpures = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatL2(gpures, dimension)
index_multi_head = []
for i in range(attention_heads):
    index_multi_head.append(faiss.GpuIndexFlatL2(gpures, head_size))
values = []

normalizer = torch.nn.functional.normalize
e = 2.7182818
added_counts = 0

def add_new(start: int, KV_path: str, of_batch: int, batch_count: int):
    n_keys, n_values = load_KV_from_binary_new(start, KV_path, of_batch, batch_count)

    index.add(np.array(n_keys).astype('float32'))

    for n_value in n_values:
        values.append(n_value)
    return

def add_new_multi_head(start: int, KV_path: str, of_batch: int, batch_count: int):
    n_keys, n_values = load_KV_from_binary_new(start, KV_path, of_batch, batch_count)
    
    for i in tqdm(range(attention_heads), 
                  total=attention_heads, 
                  desc="Loading Indexes({}/{})".format(of_batch, batch_count),
                  leave=False
                 ):
        cut = []
        for n_key in n_keys:
            cut.append(n_key[i * head_size: (i + 1) * head_size])
        
        index_multi_head[i].add(np.array(cut).astype('float32'))

    for n_value in n_values:
        values.append(n_value)
    return

def add_new_all(start: int,  KV_path: str, of_batch: int, batch_count: int):
    n_keys, n_values = load_KV_from_binary_new(start, KV_path, of_batch, batch_count)

    index.add(np.array(n_keys).astype('float32'))

    for i in tqdm(range(attention_heads), 
                  total=attention_heads, 
                  desc="Loading Indexes({}/{})".format(of_batch, batch_count),
                  leave=False
                 ):
        cut = []
        for n_key in n_keys:
            cut.append(n_key[i * head_size: (i + 1) * head_size])
        
        index_multi_head[i].add(np.array(cut).astype('float32'))

    for n_value in n_values:
        values.append(n_value)
    return

errors_prompt = []
errors_output = []
errors_answer = []

def get_next_knn(atts :list, k: int):

    dis, ind = index.search(np.array([att for att in atts]).astype('float32'), k)
    ret = []

    for ds, inds in zip(dis, ind):
        dists = []

        for i, d in enumerate(ds):
            dists.append(e ** -(d / tempreture))

        dists = normalizer(torch.tensor(dists), p=1, dim=-1)

        file = open(file_prefix + "single_ds.txt", 'w')
        print(ds, file=file)
        file.close()
        file = open(file_prefix + "single_dists.txt", 'w')
        print(dists, file=file)
        file.close()

        nxt = torch.zeros([7])
        for i in range(k):
            nxt[values[inds[i]]] = nxt[values[inds[i]]] + dists[i]
        
        ret.append(nxt)

    return ret

def get_multihead_knn(atts :list, k: int) -> torch.Tensor:

    cuts = []

    for i in range(attention_heads):
        cut = []
        for att in atts:
            cut.append(att[i * head_size: (i + 1) * head_size])
        
        dis, ind = index_multi_head[i].search(np.array(cut).astype('float32'), k)

        this_cut = []

        for ds, inds in zip(dis, ind):
            dists = []
            for d in ds:
                dists.append(e ** -(attention_heads * d / tempreture))

            dists = normalizer(torch.tensor(dists), p=1, dim=-1)
            nxt = torch.zeros([7])
            for j in range(k):
                nxt[values[inds[j]]] = nxt[values[inds[j]]] + dists[j]
            
            this_cut.append(nxt)

            if i == 1:
                file = open(file_prefix + "multi_ds.txt", 'w')
                print(ds, file=file)
                file.close()
                file = open(file_prefix + "multi_dists.txt", 'w')
                print(dists, file=file)
                file.close()
        
        cuts.append(torch.stack(this_cut, dim=0))
    
    ret = torch.stack(cuts, dim=0).permute(1, 0, 2)
    return ret

def evaluate_knn(dataloader, p: prompt_pipline, fs:int, script: str, knn_width=32):

    acc = [0, 0]
    total = 0
    # acc_multihead= torch.zeros([attention_heads])

    for data in tqdm(dataloader, total=len(dataloader), desc=script):

        prompt, cycans = p.prepare_prompt(data, ufs=fs)

        att = get_att(prompt)
        if(len(att) == 0): continue
        
        total += 1
        
        multihead_next = get_multihead_knn([att], knn_width)[0]
        # next = multihead_next.argmax(dim=-1)
        # input(next.eq(ans))
        # acc_multihead = acc_multihead + next.eq(torch.tensor(cycans).repeat(attention_heads))
        ans = torch.zeros([7])
        for i in range(attention_heads):
            ans = ans + multihead_next[i]
        acc[0] += 1 if ans.argmax(dim=0) == cycans else 0

        
        next = get_next_knn([att], knn_width)[0]
        acc[1] += 1 if next.argmax(dim=0) == cycans else 0

            # if next.argmax(dim=0) != get_data_ans(data):
            #     errors_prompt.append(prompt)
            #     errors_answer.append(get_data_ans(data))
            #     errors_output.append(next)

    return acc[0], acc[1], total
    # print(acc, total)

cpplsum1 = 0.
cpplsum2 = 0.
def evaluate_chat(dataloader, p: prompt_pipline, fs: int, script: str, full_generate=True):
    
    acc = 0
    total = 0
    human_count = 0

    for j, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=script):
        # print("{} round".format(j))

        prompt, cycans = p.prepare_prompt(data, ufs=fs)

        answer, human, cppl = get_ans(prompt, full_generate)
        if(human == -1): continue
        human_count += human
        
        acc += 1 if answer == cycans else 0
        cpplsum1 += cppl if answer == cycans else 0
        cpplsum2 += cppl if answer != cycans else 0
        
        total += 1

    return human_count, acc, total

ps = []

def chat_mode_evaluator(few_shot: list[int]):

    few_shots = len(few_shot)
    subsets_count = len(subsets)
    desc = "Testing {}-shot ({}/{}) "

    if len(ps) == 0:
        for subset in tqdm(subsets, desc="Preparing Few Shots "):
            ps.append(get_few_shot(few_shot[-1], eva_task, subset))

    print("hi")

    ffs = [0 for _ in range(few_shots)]
    accs = [0 for _ in range(few_shots)]
    tots = [0 for _ in range(few_shots)]
    for indiii, subset in enumerate(subsets):
        
        print("\n==========\nCurrently working on subset:",
              subset,
              "({}/{}).".format(indiii + 1, subsets_count))

        dataloader = get_dataloader(eva_task, subset, test_name)
        for j in range(few_shots):
            ff, acc, tot = evaluate_chat(dataloader, 
                                         ps[indiii], 
                                         few_shot[j], 
                                         desc.format(few_shot[j], j + 1, few_shots), 
                                         False)
            ffs[j] += ff
            accs[j] += acc
            tots[j] += tot
    
    with open("cppl.txt", "a") as file:
        print(cpplsum1, 
              cpplsum2, cpplsum1 / sum(accs), 
              cpplsum2 / (sum(tots) - sum(accs)), 
              file=file)
    
    file = open(file_prefix + "RESULT-baseline.txt", "a")
    for j in range(few_shots):
        print("{}-shot result:".format(few_shot[j]), file=file)
        print(ffs[j], accs[j], tots[j], 
              accs[j] / tots[j], 
              accs[j] / (tots[j] - ffs[j]), '\n', file=file)
    file.close()


def knn_evaluator(few_shot: list[int], cache_name, cache_task, cache_2_use: int, cache_2_add: list[int], topk: int):

    few_shots = len(few_shot)
    subsets_count = len(subsets)
    desc = "Testing {}-shot ({}/{}) "
    
    cache_prefix = "{}/{}/".format(cache_name, cache_task)
    cache_path = cache_task + "Cache{}_{}s/".format("_cyc" if True else "", cache_2_use)
    result_file = "RESULT-knn_Default_{}sCache.txt".format(cache_2_use)
    KV_path = cache_prefix + cache_path

    ccount = 0
    for i in cache_2_add:
        ccount += 1
        add_new_all(i, KV_path, ccount, len(cache_2_add))

    if len(ps) == 0:
        for subset in tqdm(subsets, desc="Preparing Few Shots "):
            ps.append(get_few_shot(few_shot[-1], eva_task, subset))

    acc0s = [0 for _ in range(few_shots)]
    acc1s = [0 for _ in range(few_shots)]
    tots = [0 for _ in range(few_shots)]
    for indiii, subset in enumerate(subsets):
        
        print("\n==========\nCurrently working on subset:",
              subset,
              "({}/{}).".format(indiii + 1, subsets_count))

        dataloader = get_dataloader(eva_task, subset, test_name)
        for j in range(few_shots):
            acc0, acc1, tot = evaluate_knn(dataloader, 
                                           ps[indiii], 
                                           few_shot[j], 
                                           desc.format(few_shot[j], j + 1, few_shots),
                                           topk
                                           )
            acc0s[j] += acc0
            acc1s[j] += acc1
            tots[j] += tot
    
    file = open(file_prefix + result_file, "a")
    print("\n===\n{} KV-pair entries.".format(len(values)), 
          file=file)
    for j in range(few_shots):
        print("{}-shot result:".format(few_shot[j]), file=file)
        print(acc0s[j], acc1s[j], tots[j], 
              acc0s[j] / tots[j], 
              acc1s[j] / tots[j], '\n', file=file)
    file.close()


if __name__ == "__main__":

    # knn_evaluator([0, 5], model_name, eva_task, 0, [0, 1, 2, 3, 4], 1024)
    
    chat_mode_evaluator([0, 5])
    
    save_KV_to_binary_new(logitss, [0], 0, "/data/shilh/codes/knnTest/mistral")

    # p = prompt_pipline()
    # dataloader = get_dataloader(eva_task, "sociology", test_name)
    # for data in dataloader:
    #     print("===============================")
    #     prompt, ans = p.prepare_prompt(data)
    #     print("======")
    #     print(prompt)
    #     print("======")
    #     input_ids = tokenizer(prompt, return_tensors="pt")
    #     print(input_ids["input_ids"])
    #     output, human = get_ans(prompt)
    #     print("======")
    #     print(output, human)
    #     print("======")
    #     option = model(input_ids["input_ids"].to(device))["logits"].cpu().argmax(dim=-1)[0][-5:]
    #     print(option)
    #     print(tokenizer.decode(option))
    #     input()

    # for i in range(knn_files):
    #     add_new_all(i, i + 1, knn_files)
        # accs, acc1, total1 = evaluate_knn(True, subset)
        # print(acc1, total1, file=file)
        # acc0 += acc1
        # total0 += total1
    
    # print("\n\n", acc0, total0, acc0 / total0, file=file)

    # 88 201 0.43781094527363185
    # 113 201 0.5621890547263682

    print("Done! Congratulations!")
    # errorfile = open()
    # for p, o, a in zip(errors_prompt, errors_output, errors_answer):
    #     print(p)
    #     print(o)
    #     print(a, '\n\n')
    #     input()

    # input("Done.")
'''
progress = 24                               progress = 24
save_gap = 2048                             save_gap = 2048
max_length = 768                            max_length = 768
knn_width = 32                              knn_width = 32
tempreture = 5000                           tempreture = 5000
dimension = 4096                            dimension = 4096
use_multihead = True                        use_multihead = True
attention_heads = 32                        attention_heads = 32
head_size = dimension // attention_heads    head_size = dimension // attention_heads
top_heads = 16                              top_heads = 32
use_human = False                           use_human = False

Sociology:
114 201 0.5671641791044776                  109 201 0.5422885572139303
106 201 0.527363184079602                   106 201 0.527363184079602

Human aging:
97 223 0.4349775784753363                   98 223 0.43946188340807174
101 223 0.452914798206278                   94 223 0.42152466367713004

Prehistory:
148 324 0.4567901234567901                  146 324 0.4506172839506173
112 324 0.345679012345679                   115 324 0.3549382716049383

Philosophy:
127 311 0.40836012861736337                 128 311 0.4115755627009646
126 311 0.40514469453376206                 127 311 0.40836012861736337

High school biology:
128 310 0.4129032258064516                  132 310 0.4258064516129032
111 310 0.3580645161290323                  111 310 0.3580645161290323

Econometrics:
28 114 0.24561403508771928                  31 114 0.2719298245614035
26 114 0.22807017543859648                  27 114 0.23684210526315788

Jurisprudence:
44 108 0.4074074074074074                   47 108 0.4351851851851852
48 108 0.4444444444444444                   47 108 0.4351851851851852

Marketing:
164 234 0.7008547008547008                  161 234 0.688034188034188
159 234 0.6794871794871795                  156 234 0.6666666666666666

Virology:
67 166 0.4036144578313253                   69 166 0.41566265060240964
67 166 0.4036144578313253                   65 166 0.39156626506024095

Nutrition
136 306 0.4444444444444444                  141 306 0.46078431372549017
119 306 0.3888888888888889                  116 306 0.3790849673202614

'''