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
import json
import logging
import os
import random


# Third-party library imports
import datasets
from datasets import load_dataset, load_metric, DatasetDict
import evaluate
from IPython.display import display, HTML
import jsonlines
import numpy as np
import pandas as pd
import torch
from torch import nn
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader, Dataset, IterableDataset

#.... Captum imports..................
from captum.concept import TCAV
from captum.concept import Concept
from captum.concept._utils.common import concepts_to_str

from torchtext.vocab import Vocab

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset

nlp = spacy.load('en')

# fixing the seed for CAV training purposes and performing train/test split
random.seed(123)
np.random.seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
parser.add_argument("--batch", type=int, help="Batch size")
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
    return tokenizer(examples["text"])

# Function to evaluate accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def get_tensor_from_filename(filename):
    ds = torchtext.data.TabularDataset(path=filename,
                                       fields=[('text', torchtext.data.Field()),
                                               ('label', torchtext.data.Field())],
                                       format='csv')
    const_len = 7
    for concept in ds:
        concept.text = concept.text[:const_len]
        concept.text += ['pad'] * max(0, const_len - len(concept.text))
        text_indices = torch.tensor([TEXT.vocab.stoi[t] for t in concept.text], device=device)
        yield text_indices
        
def assemble_concept(name, id, concepts_path="data/tcav/sentiment-classification"):
    dataset = CustomIterableDataset(get_tensor_from_filename, concepts_path)
    concept_iter = dataset_to_dataloader(dataset, batch_size=1)
    return Concept(id=id, name=name, data_iter=concept_iter)


def print_concept_sample(concept_iter):
    cnt = 0
    max_print = 10
    item = next(concept_iter)
    while cnt < max_print and item is not None:
        print(' '.join([TEXT.vocab.itos[item_elem] for item_elem in item[0]]))
        item = next(concept_iter)
        cnt += 1

def extract_scores(interpretations, layer_name, score_type, idx):
    return [interpretations[key][layer_name][score_type][idx].item() for key in interpretations.keys()]

def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def plot_tcav_scores(experimental_sets, tcav_scores, layers = ['model.deberta.encoder.output', 'model.linear'], score_type='sign_count', fig_path = "plot.png"):
    fig, ax = plt.subplots(1, len(experimental_sets), figsize = (25, 7))

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):
        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)
        
        layers = tcav_scores[concepts_key].keys()
        pos = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i-1]])
        _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
        for i in range(len(concepts)):
            val = [format_float(scores[score_type][i]) for layer, scores in tcav_scores[concepts_key].items()]
            _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

        # Add xticks on the middle of the group bars
        _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
        _ax.set_xticklabels(layers, fontsize=16)

        # Create legend & Show graphic
        _ax.legend(fontsize=16)

    plt.savefig(fig_path)




#######################################
#             Main code               #
#######################################


neutral_concept = assemble_concept('neutral', 0, concepts_path="data/tcav/sentiment-classification/neutral.csv")
positive_concept = assemble_concept('positive-adjectives', 5, \
                                    concepts_path="data/tcav/sentiment-classification/positive-adjectives.csv")

neutral_concept = assemble_concept('neutral', 0, concepts_path="data/tcav/sentiment-classification/neutral.csv")
neutral_concept2 = assemble_concept('neutral2', 1, concepts_path="data/tcav/sentiment-classification/neutral2.csv")
neutral_concept3 = assemble_concept('neutral3', 2, concepts_path="data/tcav/sentiment-classification/neutral3.csv")
neutral_concept4 = assemble_concept('neutral4', 3, concepts_path="data/tcav/sentiment-classification/neutral4.csv")
neutral_concept5 = assemble_concept('neutral5', 4, concepts_path="data/tcav/sentiment-classification/neutral5.csv")

positive_concept = assemble_concept('positive-adjectives', 5, \
                                    concepts_path="data/tcav/sentiment-classification/positive-adjectives.csv")

print(f"Printing concept samples . . . . . ")
print_concept_sample(iter(positive_concept.data_iter))

experimental_sets=[[positive_concept, neutral_concept],
                  [positive_concept, neutral_concept2],
                  [positive_concept, neutral_concept3],
                  [positive_concept, neutral_concept4],
                  [positive_concept, neutral_concept5]]


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
# format = {'type': 'torch', 'format_kwargs' :{'dtype': torch.float}}
# dataset.set_format(**format)

# dataset.set_format(output_all_columns=True, type='numpy', columns=['label'], format_kwargs={"dtype":int})
# Set evaluation metrics from HF evaluate to "accuracy"
accuracy_metric = evaluate.load("accuracy")



# Preprocess the dataset using HF's tokenizer using the Fast tokenizer

model_checkpoint = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True) 



# Convert tokens to token_ids
logger.info(f"Tokenizing data . . . . ")
encoded_dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)
encoded_dataset.set_format(output_all_columns=True, type='numpy', columns=['label'], format_kwargs={"dtype":float})
# Specify sequence classification model and its config
num_labels = 2

# Specify trainer + args + hyper-params
metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]
task="imdb_classification"
batch_size = 4
validation_key = "validation"

output_dir_path = os.path.join(args.output_dir, f"{model_name}-finetuned-{task}")

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


tcav = TCAV(model, layers=['model.deberta.encoder.output', 'model.linear'])

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


plot_tcav_scores(experimental_sets, encoded_dataset["test"], ['model.deberta.encoder.output', 'model.linear'], score_type='sign_count')


############################################################################
#       Older code (Not strictly required)
############################################################################
# Evaluate the trainer model  
predictions, labels, final_metrics = trainer.predict(encoded_dataset["test"])
print(f"Final Test metrics:{final_metrics}")
outputs = []

predictions = np.argmax(predictions, axis=1)

with open(f"{output_dir_path}/test_outputs.json", "w") as op_file:
    for data_idx, data_sample in enumerate(dataset["test"]):
        output = {
                "review": data_sample["text"],
                "label": int(labels[data_idx]),
                "predicted": int(predictions[data_idx]),
            }
        outputs.append(
            output
        )
        print(f"output:{output}")
        print(f"op_label:{output['label']}")
        print(f"op_predicted:{output['predicted']}")
    json.dump(outputs, op_file)

