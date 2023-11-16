from scorer import QAScore
from datasets import load_dataset
import pickle
import os

device = 'cuda:2'
dataset = load_dataset('squad')['validation']

qascore = QAScore(device=device)
res= qascore.score_by_dataset(paragraphs=dataset['context'], questions=dataset['question'], answers=dataset['answers'], use_tqdm=True)
with open('QAScore_SQuAD.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
print("mean QAScore: ", res[1])