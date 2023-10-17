from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from MyDataset import MyDataset_question_generation
from evaluate import load

squad = load_dataset("squad")

tokenizer = AutoTokenizer.from_pretrained("/storagenfs/m.tolloso/HLT_Project/models/t5-large/checkpoint-1000")
model = AutoModelForSeq2SeqLM.from_pretrained("/storagenfs/m.tolloso/HLT_Project/models/t5-large/checkpoint-1000")

tokenized_squad_train = MyDataset_question_generation(squad["train"], tokenizer)
tokenized_squad_val = MyDataset_question_generation(squad["validation"], tokenizer)

rqugescore = load("alirezamsh/rquge")

for i in range(len(squad["validation"])):
    tokenized_input = tokenized_squad_val[i]
    output = model.generate(tokenized_input['input_ids'].reshape(1, -1), num_beams=2, max_length=200,
                                         decoder_start_token_id=tokenizer.pad_token_id)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=False)
    print(squad["validation"][i]["context"])
    print(squad["validation"][i]["answers"]["text"])
    print(decoded_output)
    print(squad["validation"][i]["question"])

    results = rqugescore.compute(generated_questions=decoded_output,
                                 contexts=squad["validation"][i]["context"],
                                 answers=squad["validation"][i]["answers"]["text"])

    print(results["mean_score"])
    print()
    input()

