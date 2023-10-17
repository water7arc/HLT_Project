from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, EarlyStoppingCallback
from MyDataset import MyDataset_answer_generation


squad = load_dataset("squad")

tokenizer = AutoTokenizer.from_pretrained("/storagenfs/m.tolloso/HLT_Project/models/t5-small_asnwergen_overfit/checkpoint-3000")
model = AutoModelForQuestionAnswering.from_pretrained("/storagenfs/m.tolloso/HLT_Project/models/t5-small_asnwergen_overfit/checkpoint-3000")

tokenized_squad_train = MyDataset_answer_generation(squad["train"], tokenizer)
tokenized_squad_val = MyDataset_answer_generation(squad["validation"], tokenizer)

for i in range(100):
    tokenized_input = tokenized_squad_train[i]
    output = model(**tokenized_input)
    answer_start_index = output.start_logits.argmax()
    answer_end_index = output.end_logits.argmax()
    predict_answer_tokens = tokenized_input.input_ids[0, answer_start_index: answer_end_index + 1]
    decoded_output = tokenizer.decode(predict_answer_tokens)

    print(squad["train"][i]["context"])
    print(squad["train"][i]["answers"]["text"])
    print(decoded_output)
    print(squad["train"][i]["question"])
    print()
    input()

