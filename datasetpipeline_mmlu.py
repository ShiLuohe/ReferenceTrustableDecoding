question_head = '''Choose the best answer of the question. Output A, B, C, etc.
'''
question_format = '''Question: {}
Options: {}
The answer is: {}
'''

import torch

from datasets import load_dataset

eva_task = "mmlu"
subsets =  ['abstract_algebra', 'anatomy', 'astronomy', 
            'business_ethics', 'clinical_knowledge', 'college_biology', 
            'college_chemistry', 'college_computer_science', 'college_mathematics', 
            'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 
            'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 
            'global_facts', 'high_school_biology', 'high_school_chemistry', 
            'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 
            'high_school_government_and_politics', 'high_school_macroeconomics', 
            'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 
            'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 
            'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 
            'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 
            'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 
            'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 
            'professional_medicine', 'professional_psychology', 'public_relations', 
            'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

train_name = "auxiliary_train"
vali_name = "validation"
test_name = 'test'

# =============================================================
def get_dataloader(name: str, subset: str, split: str):
    dataset = load_dataset(name, subset, split=split)
    dataset.set_format(type="torch", columns=["question", "choices", "answer"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    return dataloader

# =============================================================
answer_convert = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
                  '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5,
                   1 : 0,  2 : 1,  3 : 2,  4 : 3,  5 : 4,  6 : 5}
def get_data_answer(data):
    return data["answer"][0].tolist()

def get_data_lines(data) -> tuple[str, list[str]]:
    question_prompt = data["question"][0]
    choices_prompts = []
    for choice in data["choices"]:
        choices_prompts.append(choice[0])
    return question_prompt, choices_prompts

# =============================================================
def view_dataset():
    while True:
        dataset_name = input("Dataset name : ")
        dataset_split = input("Dataset split: ")

        dataloader = get_dataloader("mmlu", dataset_name, dataset_split)

        for data in dataloader:
            print(get_data_lines(data), "\n===")
            print(get_data_answer(data))
            input()

if __name__ == "__main__":

    view_dataset()