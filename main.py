import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from MyDataset import MyDataset


squad = load_dataset("squad", split="train[:100]")
squad = squad.train_test_split(test_size=0.2, shuffle=True, seed=42)

# tokenizer = AutoTokenizer.from_pretrained("t5-small")
# model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-small")

# tokenizer = AutoTokenizer.from_pretrained("models/t5-fine_tune-overfit/checkpoint-2000")
# model = AutoModelForSeq2SeqLM.from_pretrained("models/t5-fine_tune-overfit/checkpoint-2000")

# def preprocess_function(examples):
#     questions = [q.strip() for q in examples["question"]]
#
#     for i in range(len(examples["context"])):
#         context = examples["context"][i]
#         answer_start = examples["answers"][i]["answer_start"][0]
#         answer_end = answer_start + len(examples["answers"][i]["text"][0])
#         examples["context"][i] = context[:answer_start] + \
#                                  "</s>" + context[answer_start:answer_end] + "</s>" + context[answer_end:]
#     inputs = tokenizer(
#         examples["context"],
#         text_target=questions,
#         max_length=384,
#         truncation="only_first",
#         padding="max_length",
#         return_tensors="pt",
#     )
#
#     inputs["labels"][inputs["labels"] == tokenizer.pad_token_id] = -100
#
#     return inputs

# for i in range(len(squad["train"])):
#     tokenized_in = preprocess_function(squad["train"][i:i+1])
#     tokenized_in["labels"][tokenized_in["labels"] == -100] = tokenizer.pad_token_id
#     output = model.generate(tokenized_in['input_ids'], num_beams=2, max_length=200,
#                                          decoder_start_token_id=tokenizer.pad_token_id)
#     decoded_output = tokenizer.batch_decode(output, skip_special_tokens=False)
#     print(squad["train"][i]["context"])
#     print(squad["train"][i]["answers"]["text"])
#     print(decoded_output)
#     print(squad["train"][i]["question"])

# tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

tokenized_squad_train = MyDataset(squad["train"], tokenizer)
tokenized_squad_val = MyDataset(squad["test"], tokenizer)
# text = tokenizer.decode(tokenized_squad["train"][0]["input_ids"])
# print(text)
# question = tokenizer.decode(tokenized_squad["train"][0]["labels"])
# print(question)


data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="models/t5-fine_tune-overfit/",
    evaluation_strategy="epoch",
    learning_rate=2e-3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=50,
    weight_decay=0.001,
    push_to_hub=False,
    report_to=["wandb"],
    # no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad_train,
    eval_dataset=tokenized_squad_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
