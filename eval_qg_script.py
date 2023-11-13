from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from MyDataset import MyDataset_question_generation
from evaluate import load
import pickle
from tqdm import tqdm


BATCH_SIZE = 50

def evaluate_qg_models(model_names, dataset):
    results = {}
    for model_name in model_names:
        model_output = []
        tokenizer = AutoTokenizer.from_pretrained(QGPATH + model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(QGPATH + model_name).to("cuda")

        tokenized_squad_val = MyDataset_question_generation(dataset, tokenizer)

        for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc=model_name):
            contexts = dataset[i:i+BATCH_SIZE]["context"]
            answers = [a["text"][0] for a in dataset[i:i+BATCH_SIZE]["answers"]]
            reference_questions = dataset[i:i+BATCH_SIZE]["question"]

            tokenized_input = tokenized_squad_val[i:i+BATCH_SIZE]
            output = model.generate(tokenized_input['input_ids'].to("cuda"), num_beams=2, max_length=200,
                                    decoder_start_token_id=tokenizer.pad_token_id)
            decoded_outputs = tokenizer.batch_decode(output, skip_special_tokens=False)
            generated_questions = [output.strip('<pad>')[1:-3] for output in decoded_outputs]

            for context, answer, reference_question, generated_question in zip(contexts, answers, reference_questions, generated_questions):
                model_output.append({"context": context,
                                    "answer": answer,
                                    "reference_question": reference_question,
                                    "generated_question": generated_question
                                    })

        with open("model_output", "wb") as f:
            pickle.dump(model_output, f)
        # with open("model_output", "rb") as f:
        #     model_output = pickle.load(f)

        generated_questions_batch = [result["generated_question"] for result in model_output]
        reference_questions_batch = [result["reference_question"] for result in model_output]
        contexts = [result["context"] for result in model_output]
        answers = [result["answer"] for result in model_output]

        bleu_score = bleu_metric.compute(predictions=generated_questions_batch, references=reference_questions_batch)
        rouge_score = rouge_metric.compute(predictions=generated_questions_batch, references=reference_questions_batch)
        rquge_score = rquge_metric.compute(generated_questions=generated_questions_batch, contexts=contexts, answers=answers, device="cuda")

        results[model_name] = {"output": model_output,
                               "bleu": bleu_score,
                               "rouge": rouge_score,
                               "rquge": rquge_score
                               }

    return results


squad = load_dataset("squad")
bleu_metric = load("bleu")
rouge_metric = load("rouge")
rquge_metric = load("alirezamsh/rquge")

QGPATH = "../../../m.tolloso/HLT_Project/models/QuestionGen/"

qg_models = {"t5-tiny"}

results = evaluate_qg_models(qg_models, squad["validation"])

with open("results", "wb") as f:
    pickle.dump(results, f)

# with open("results", "rb") as f:
#     results = pickle.load(f)
#     print()