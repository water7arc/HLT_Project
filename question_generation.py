from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, EarlyStoppingCallback
from MyDataset import MyDataset_question_generationGPT2
import wandb

squad = load_dataset("squad")

model_name = "gpt2"
run_name = "gpt2"
path_name = "models/" + run_name
prj_name = "huggingface"

run = wandb.init(name=run_name, project=prj_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenized_squad_train = MyDataset_question_generationGPT2(squad["train"], tokenizer)
tokenized_squad_val = MyDataset_question_generationGPT2(squad["validation"], tokenizer)


data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir=path_name,
    evaluation_strategy="steps",
    learning_rate=1e-3,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=22,
    per_device_eval_batch_size=22,
    num_train_epochs=50,
    push_to_hub=False,
    report_to=["wandb"],
    save_steps=1000,
    save_total_limit=5,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end = True,
    use_mps_device=True,
    optim= "adamw_torch",
    # no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad_train,
    eval_dataset=tokenized_squad_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()