question_head = '''选择问题的最佳选项。用A B C D作答。
'''
question_format = '''问题: {}
选项: {}
答案: {}
'''

import torch
from datasets import load_dataset

eva_task = "cmmlu"
subsets = ["agronomy", "test"]

train_name = "dev"
vali_name = "dev"
test_name = 'test'

# =============================================================
def get_dataloader(name: str, subset: str, split: str):
    dataset = load_dataset("cmmlu", subset, split=split)
    dataset.set_format(type="torch", columns=["Question", "A",
                                                          "B",
                                                          "C", 
                                                          "D", "Answer"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    return dataloader

# =============================================================
answer_convert = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
                  '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5,
                   1 : 0,  2 : 1,  3 : 2,  4 : 3,  5 : 4,  6 : 5}
def get_data_answer(data) -> int:
    return answer_convert[data["Answer"][0]]

def get_data_lines(data) -> tuple[str, list[str]]:
    question_prompt = data["Question"][0]
    choices_prompts = [data["A"][0], data["B"][0], data["C"][0], data["D"][0]]
    return question_prompt, choices_prompts

# =============================================================
def view_dataset():
    while True:
        dataset_name = input("Dataset name : ")
        dataset_split = input("Dataset split: ")

        dataloader = get_dataloader("ai2_arc", dataset_name, dataset_split)

        for data in dataloader:
            print(get_data_lines(data), "\n===")
            print(get_data_answer(data))

if __name__ == "__main__":

    view_dataset()