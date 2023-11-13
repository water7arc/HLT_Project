from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, EarlyStoppingCallback
from MyDataset import MyDataset_answer_generation, MyDataset_question_generation
from transformers import AutoModelForQuestionAnswering
from evaluate import load
import pickle
from tqdm import tqdm


BATCH_SIZE = 50


def evaluate_ag_models(model_names, dataset):
    results = {}
    for model_name in model_names:
        model_output = []
        tokenizer = AutoTokenizer.from_pretrained(AGPATH + model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(AGPATH + model_name).to("cuda")

        dataset = process_dataset_for_ag(dataset)
        tokenized_squad_val = MyDataset_answer_generation(dataset, tokenizer)

        for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc=model_name):
            contexts = dataset[i:i+BATCH_SIZE]["context"]
            reference_answers = [a["text"] for a in dataset[i:i+BATCH_SIZE]["answers"]]
            questions = dataset[i:i+BATCH_SIZE]["question"]

            tokenized_input = tokenized_squad_val[i:i+BATCH_SIZE]
            output = model.generate(tokenized_input['input_ids'].to("cuda"), num_beams=10, num_return_sequences=5,
                                    max_length=15, decoder_start_token_id=tokenizer.pad_token_id)
            decoded_outputs = tokenizer.batch_decode(output, skip_special_tokens=False)
            generated_answers = [[output.strip('<pad>')[1:-3] for output in decoded_outputs[i:i+5]] for i in range(0, BATCH_SIZE*5, 5)]

            for context, reference_answer, question, generated_answer in zip(contexts, reference_answers, questions, generated_answers):
                model_output.append({"context": context,
                                    "reference_answer": reference_answer,
                                    "question": question,
                                    "generated_answer": generated_answer
                                    })

        with open("model_output_ag", "wb") as f:
            pickle.dump(model_output, f)

        generated_answer_batch = []
        reference_answer_batch = []

        for result in model_output:
            for answer in result["generated_answer"]:
                generated_answer_batch.append(answer)
                reference_answer_batch.append(result["reference_answer"])

        bleu_score = bleu_metric.compute(predictions=generated_answer_batch, references=reference_answer_batch)
        rouge_score = rouge_metric.compute(predictions=generated_answer_batch, references=reference_answer_batch)

        results[model_name] = {"output": model_output,
                               "bleu": bleu_score,
                               "rouge": rouge_score,
                               }

    return results


def process_dataset_for_ag(dataset):
    already_seen_context = {}
    context_index = []

    for i in range(len(dataset)):
        context = dataset[i]["context"]
        if context not in already_seen_context.keys():
            context_index.append(i)
            already_seen_context[context] = {"questions": [], "answers": []}
        already_seen_context[context]["questions"].append(dataset[i]["question"])
        already_seen_context[context]["answers"].append(dataset[i]["answers"]["text"][0])
    
    dataset = dataset.select(context_index)

    def answer_map(sample):
        sample["answers"]["text"] = already_seen_context[sample["context"]]["answers"]
        sample["question"] = already_seen_context[sample["context"]]["questions"]
        return sample
    dataset = dataset.map(answer_map)
    return dataset



squad = load_dataset("squad")
bleu_metric = load("bleu")
rouge_metric = load("rouge")
rquge_metric = load("alirezamsh/rquge")


AGPATH = "../../../m.tolloso/HLT_Project/models/AnswerGen/"

ag_models = {"t5-small_answergen"}

results = evaluate_ag_models(ag_models, squad["validation"])

with open("results", "wb") as f:
    pickle.dump(results, f)