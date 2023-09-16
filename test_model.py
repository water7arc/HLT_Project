import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, EarlyStoppingCallback
from MyDataset import MyDataset


squad = load_dataset("squad")

tokenizer = AutoTokenizer.from_pretrained("models/checkpoint-25000")
model = AutoModelForSeq2SeqLM.from_pretrained("models/checkpoint-25000")

tokenized_squad_train = MyDataset(squad["train"], tokenizer)
tokenized_squad_val = MyDataset(squad["validation"], tokenizer)

for i in range(len(squad["train"])):
    tokenized_input = tokenized_squad_train[i]
    output = model.generate(tokenized_input['input_ids'].reshape(1, -1), num_beams=2, max_length=200,
                                         decoder_start_token_id=tokenizer.pad_token_id)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=False)
    print(squad["train"][i]["context"])
    print(squad["train"][i]["answers"]["text"])
    print(decoded_output)
    print(squad["train"][i]["question"])
    print()

