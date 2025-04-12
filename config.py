
from datasetpipeline_mmlu import *
from model_Mistral import *

# ==== Evaluation argumants ====
class evaluation_args:
    def __init__(self, prog=0, knn_no=2, fwst=0, knn_w=32, temp=1000) -> None:
        self.subset_progress = prog
        self.knn_files = knn_no
        self.knn_width = knn_w
        self.tempreture = temp

# subset_progress = 0
# # How many subsets have done
# knn_files = 2
# # Amont of cached knn data to load
# few_shots = 5
# # Few shots validation
# knn_width = 32
# # How many knn cases to be considered
# tempreture = 1000
# # Normalization tempreture

# ==== Cache generation arguments ====
max_generation_amount = 4096 * 5
# Max generation amount
save_gap = 4096
# Size of cached knn data
use_cycle = True
# Use cycle generation
progress = 0
# Amount of cached file that has been generated
exists = 0
# Amount of cached file that has existed
subset_no_to_use = 0
# Which subset to use to generate cache
use_selected = False
# Use selected attention as cache

# ==== Cache loading arguments ====
# cache_fwst = 5
# cache_ucyc = True
# cache_uslc = False

file_prefix = "{}/{}/".format(model_name, eva_task)
# KV_file = "{}Cache{}{}_{}s/".format(cache_task, 
#                                          "_cyc" if cache_ucyc else "",
#                                          "_slc" if cache_uslc else "", 
#                                          cache_fwst)
# rounds = 0
# raw_file_path = file_prefix + "{}_scores_{}s_{}.txt".format(eva_task, few_shots, rounds)
# analysied_file_path = file_prefix + "{}_ana_{}s_{}.txt".format(eva_task, few_shots, rounds)