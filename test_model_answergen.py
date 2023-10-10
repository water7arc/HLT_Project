import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, EarlyStoppingCallback
from MyDataset import MyDataset_answer_generation


squad = load_dataset("squad")

tokenizer = AutoTokenizer.from_pretrained("/storagenfs/m.tolloso/HLT_Project/models/t5-base_asnwergen/checkpoint-6000")
model = AutoModelForSeq2SeqLM.from_pretrained("/storagenfs/m.tolloso/HLT_Project/models/t5-base_asnwergen/checkpoint-6000")

tokenized_squad_train = MyDataset_answer_generation(squad["train"], tokenizer)
tokenized_squad_val = MyDataset_answer_generation(squad["validation"], tokenizer)

for i in range(len(squad["validation"])):
    tokenized_input = tokenized_squad_val[i]
    output = model.generate(
        tokenized_input['input_ids'].reshape(1, -1), 
        num_beams=10, 
        max_length=10,
        decoder_start_token_id=tokenizer.pad_token_id,
        num_return_sequences=10
    )
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=False)
    print(squad["validation"][i]["context"])
    print(squad["validation"][i]["answers"]["text"])
    print(decoded_output)
    print(squad["validation"][i]["question"])
    print()
    input()

