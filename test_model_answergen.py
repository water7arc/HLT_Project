import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, EarlyStoppingCallback
from MyDataset import MyDataset_answer_generation


squad = load_dataset("squad")

tokenizer = AutoTokenizer.from_pretrained("/storagenfs/m.tolloso/HLT_Project/models/t5-small_asnwergen_overfit/checkpoint-3000")
model = AutoModelForSeq2SeqLM.from_pretrained("/storagenfs/m.tolloso/HLT_Project/models/t5-small_asnwergen_overfit/checkpoint-3000")

tokenized_squad_train = MyDataset_answer_generation(squad["train"], tokenizer)
tokenized_squad_val = MyDataset_answer_generation(squad["validation"], tokenizer)

for i in range(100):
    tokenized_input = tokenized_squad_train[i]
    output = model.generate(
        tokenized_input['input_ids'].reshape(1, -1), 
        num_beams=20, 
        # max_length=10,
        decoder_start_token_id=tokenizer.pad_token_id,
        num_return_sequences=10
    )
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=False)
    print(squad["train"][i]["context"])
    print(squad["train"][i]["answers"]["text"])
    print(decoded_output)
    print(squad["train"][i]["question"])
    print()
    input()

