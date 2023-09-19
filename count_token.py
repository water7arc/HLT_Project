import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, EarlyStoppingCallback
from MyDataset import MyDataset
import numpy as np
import pickle

squad = load_dataset("squad")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

tokenized_squad = MyDataset(squad['train'], tokenizer)

lens = []
for i in range(len(tokenized_squad)):
    lens.append(np.count_nonzero(tokenized_squad[i]['attention_mask']))

print(np.mean(lens))

with open("token_lens", "wb") as f:
    pickle.dump(lens,f)
