from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


squad = load_dataset("squad")

tokenizer = AutoTokenizer.from_pretrained("models/t5-small-first-test/checkpoint-12000")
model = AutoModelForSeq2SeqLM.from_pretrained("models/t5-small-first-test/checkpoint-12000")


