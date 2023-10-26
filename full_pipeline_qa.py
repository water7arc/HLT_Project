import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, EarlyStoppingCallback
from MyDataset import MyDataset_answer_generation
from transformers import AutoModelForQuestionAnswering
from evaluate import load


squad = load_dataset("squad")
contexts = set()
for item in squad['validation']:
    contexts.add(item['context'])

tokenizer_ag = AutoTokenizer.from_pretrained("/storagenfs/m.tolloso/HLT_Project/models/google/bert-base-uncased_answergen_qa/checkpoint-4000")
model_ag = AutoModelForQuestionAnswering.from_pretrained("/storagenfs/m.tolloso/HLT_Project/models/google/bert-base-uncased_answergen_qa/checkpoint-4000")

tokenizer_qg = AutoTokenizer.from_pretrained("/storagenfs/m.tolloso/HLT_Project/models/t5-base_answer_begin/checkpoint-3000")
model_qg = AutoModelForSeq2SeqLM.from_pretrained("/storagenfs/m.tolloso/HLT_Project/models/t5-base_answer_begin/checkpoint-3000")

rqugescore = load("alirezamsh/rquge")


for context in contexts:
    tokenized_in_ag = tokenizer_ag(
            context,
            max_length=256,
            truncation="only_first",
            # return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

    output = model_ag(**tokenized_in_ag)
    answer_start_index = output.start_logits.argmax()
    answer_end_index = output.end_logits[0, answer_start_index+1:].argmax()
    answer_end_index += answer_start_index + 1
    predict_answer_tokens = tokenized_in_ag["input_ids"][0, answer_start_index: answer_end_index + 1]
    answers = tokenizer_ag.decode(predict_answer_tokens)

    answers = [answers]

    for answer in answers:

        # new_context = answer + "</s>" +  context 
        new_context = context 

        tokenized_in_qg = tokenizer_qg(
            new_context,
            # text_target='',
            max_length=256,
            truncation="only_first",
            padding="max_length",
            return_tensors="pt",
        )

        input_ids_qg = tokenized_in_qg["input_ids"][0]
        
        output = model_qg.generate(input_ids_qg.reshape(1, -1), num_beams=2, max_length=200,
                                            decoder_start_token_id=tokenizer_qg.pad_token_id)
        question = tokenizer_qg.batch_decode(output, skip_special_tokens=False)
        question = question[0].strip('<pad>')[1:-3]

        print(context, "\n")
        print(question)
        print(answer)

        # results = rqugescore.compute(generated_questions=[question],
        #                         contexts=[context],
        #                         answers=[answer], device="cuda")
        
        # print(results["mean_score"])

    input()

