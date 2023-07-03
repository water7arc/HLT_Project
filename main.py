from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer


squad = load_dataset("squad", split="train[:5000]")
squad = squad.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        examples["context"],
        text_target=questions,
        max_length=384,
        truncation="only_first",
        return_offsets_mapping=True,
        padding="max_length",
    )

    # inputs["labels"] = [[n if n != tokenizer.pad_token_id else -100 for n in l] for l in inputs["labels"]]

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 0:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 0:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_position = 0
            end_position = 0
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_position = idx - 1

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_position = idx + 1

        inputs["input_ids"][i].insert(start_position, tokenizer.eos_token_id)
        inputs["input_ids"][i].insert(end_position + 2, tokenizer.eos_token_id)
        inputs["attention_mask"][i].insert(start_position, 1)
        inputs["attention_mask"][i].insert(end_position + 2, 1)

    return inputs


tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
# text = tokenizer.decode(tokenized_squad["train"][0]["input_ids"])
# print(text)
# question = tokenizer.decode(tokenized_squad["train"][0]["labels"])
# print(question)


data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="my_awesome_qa_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    report_to=["wandb"]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
