"""
Author: Gurunath Parasaram
Course: CS 6966 - Local Explanations for Deep Learning by Prof. Ana Marasović

Acknowledgements:
This script has been loosely based on the Text Classification tutorial from Huggingface
here: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb#scrollTo=s_AY1ATSIrIq
and also other Huggingface resources.
"""

# Imports

# Standard library imports
import argparse
import logging
import os
import random


# Third-party library imports
import datasets
from datasets import load_dataset, load_metric, DatasetDict
import evaluate
from IPython.display import display, HTML
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
args = parser.parse_args()

# Set logging config
logging.basicConfig(filename='train.log', filemode='w', level=logging.INFO, format='%(process)d-%(levelname)s-%(message)s')
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

#######################################
#             Helper functions        #
#######################################

# Helper functions at the start
# Helper function to display random elements
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    # display(HTML(df.to_html()))

# Function to tokenize using HF's tokenizer
def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, return_tensors="pt", truncation=True)

# Function to evaluate accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


#######################################
#             Main code               #
#######################################

# Get IMDB dataset from HF
logger.info(f"Loading IMDB dataset")
dataset = load_dataset("imdb")

# REFERENCE: https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090/13

# IMDB dataset just has train:test splits in 25k:25k ratio
# Create validation set from training set
# Split train test into train+validation sets in 80:20 ratio
logger.info(f"Split train test into train+validation sets in 80:20 ratio")
train_plus_validation = dataset["train"].train_test_split(test_size=0.2)
dataset = DatasetDict({
    'train': train_plus_validation['train'],
    'validation': train_plus_validation['test'],
    'test': dataset["test"],
})
format = {'type': 'torch', 'format_kwargs' :{'dtype': torch.float}}
dataset.set_format(**format)

# Set evaluation metrics from HF evaluate to "accuracy"
accuracy_metric = evaluate.load("accuracy")



# Preprocess the dataset using HF's tokenizer using the Fast tokenizer

model_checkpoint = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True) 



# Convert tokens to token_ids
logger.info(f"Tokenizing data . . . . ")
encoded_dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)

# Specify sequence classification model and its config
num_labels = 1

# Specify trainer + args + hyper-params
metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]
task="imdb_classification"
batch_size = 2
validation_key = "validation"

output_dir_path = os.path.join(args.output_dir, f"{model_name}-finetuned-{task}")

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

args = TrainingArguments(
    output_dir_path,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
)

logger.info(f"Training args:{args}")
# Piggyback on HF's Trainer code
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

logger.info(f"Starting training")
# Finally...train the model!
trainer.train()

# Evaluate the trainer model  
trainer.evaluate()


