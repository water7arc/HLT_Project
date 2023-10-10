import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, EarlyStoppingCallback
from MyDataset import MyDataset_answer_generation
import numpy as np
import pickle
from tqdm import tqdm

squad = load_dataset("squad")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

tokenized_squad = MyDataset_answer_generation(squad['train'], tokenizer)

lens = []
for i in tqdm(range(len(tokenized_squad)//3), desc='Process'):
    lens.append(np.count_nonzero(tokenized_squad[i]['labels']))
    if lens[i] > 34:
        print(squad['train'][i]['answers'])

print(lens)
print(np.std(lens))

with open("token_lens", "wb") as f:
    pickle.dump(lens,f)
