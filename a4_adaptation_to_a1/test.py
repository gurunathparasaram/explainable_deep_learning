import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

model_checkpoint = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True) 

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

print(f"model params:{model}")
