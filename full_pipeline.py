import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, EarlyStoppingCallback
from MyDataset import MyDataset_answer_generation
from transformers import AutoModelForQuestionAnswering
from evaluate import load
from csv import writer
import pandas as pd


squad = load_dataset("squad")

contexts = set()
for item in squad['validation']:
    contexts.add(item['context'])

QGPATH = "./models/QuestionGen/"
AGPATH = "./models/AnswerGen/"

qg_models = {"mt5_small_qg_en", "t5-base", "t5-base_answer_begin", "t5-large", "t5-large_answer_begin",
             "t5_qg_pretrainedSquad2", "t5-small", "t5-small_answer_begin", "t5-small_answer_begin_bigbatch",
             "t5-small-repetition", "t5-tiny"}

ag_models = {"t5-base_asnwergen", "t5-base_asnwergen_short", "t5-small_answergen", "t5-small_asnwergen_short"}


tokenizer_ag = AutoTokenizer.from_pretrained()
model_ag = AutoModelForSeq2SeqLM.from_pretrained()

tokenizer_qg = AutoTokenizer.from_pretrained()
model_qg = AutoModelForSeq2SeqLM.from_pretrained()

rqugescore = load("alirezamsh/rquge")

if not os.path.exists('results.csv'):
    with open('results.csv', 'w') as f_object:
        writer_object = writer(f_object)
        title = ['QGmodel', 'AGmodel', 'context', 'answer', 'question', 'rquge_score']
        writer_object.writerow(title)
        f_object.close()
else:
    results = pd.read_csv('results.csv')
    computedQGModels = results['QGmodel'].unique()
    computedAGModels = results['AGmodel'].unique()
    computedContexts = results['context'].unique()[:-1]
    computedAnswers = results['answer'].unique()
    
    
ResultsObj = open('results.csv', 'a')
Results_Writer = writer(ResultsObj)

for QuestionGenModel in qg_models:
    
    if QuestionGenModel in computedQGModels:
        continue
    computedQGModels.add(QuestionGenModel)
    
    tokenizer_qg = AutoTokenizer.from_pretrained(QuestionGenModel)
    model_qg = AutoModelForSeq2SeqLM.from_pretrained(QuestionGenModel)
    
    for AnswerGenModel in ag_models:
        
        if AnswerGenModel in computedAGModels:
            continue
        computedAGModels.add(AnswerGenModel)
        
        tokenizer_ag = AutoTokenizer.from_pretrained(AnswerGenModel)
        model_ag = AutoModelForSeq2SeqLM.from_pretrained(AnswerGenModel)
        for context in contexts:
            
            if context in computedContexts:
                continue
            computedContexts.add(context)
            
            tokenized_in_ag = tokenizer_ag(
                    context,
                    max_length=256,
                    truncation="only_first",
                    padding="max_length",
                    return_tensors="pt",
                )
            
            input_ids_ag = tokenized_in_ag["input_ids"][0]

            output = model_ag.generate(
                input_ids_ag.reshape(1, -1), 
                num_beams=20, 
                decoder_start_token_id=tokenizer_ag.pad_token_id,
                num_return_sequences=10
            )
            answers = tokenizer_ag.batch_decode(output, skip_special_tokens=False)
            answers = [ans.strip('<pad>')[1:-3] for ans in answers]

            for answer in answers:
                if answer in computedAnswers:
                    continue
                computedAnswers.add(answer)
                
                new_context = answer + "</s>" +  context 

                tokenized_in_qg = tokenizer_qg(
                    new_context,
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
                
                results = rqugescore.compute(generated_questions=[question],
                                        contexts=[context],
                                        answers=[answer])
                
                Results_Writer.writerow([model_qg.name, model_ag.name, context, answer, question, results['rquge']])
                
                