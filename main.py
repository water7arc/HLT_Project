import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, EarlyStoppingCallback
from MyDataset import MyDataset


squad = load_dataset("squad")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

tokenized_squad_train = MyDataset(squad["train"], tokenizer)
tokenized_squad_val = MyDataset(squad["validation"], tokenizer)


data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="models/t5-base-fine_tune/",
    evaluation_strategy="steps",
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    num_train_epochs=50,
    push_to_hub=False,
    report_to=["wandb"],
    save_steps=1000,
    save_total_limit=5,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end = True,
    # no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad_train,
    eval_dataset=tokenized_squad_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.train()