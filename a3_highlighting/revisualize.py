import torch
import pandas as pd

from IPython.core.display import display, HTML
from torch import tensor 
import transformers
from transformers.pipelines import TextClassificationPipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

import matplotlib.pyplot as plt

import argparse 
import jsonlines
import os 


def show_text_attr(attrs, tokenized_text):
    print(f"attrs:{attrs}")
    rgb = lambda x: '255,0,0' if x < 0 else '0,255,0'
    alpha = lambda x: abs(x) ** 0.5
    token_marks = [
        f'<mark style="background-color:rgba({rgb(attr)},{alpha(attr)})">{token}</mark>'
        for token, attr in zip(tokenized_text, attrs)
    ]
    
    return HTML('<p>' + ' '.join(token_marks) + '</p>')
    

class ExplainableTransformerPipeline():
    """Wrapper for Captum framework usage with Huggingface Pipeline"""
    
    def __init__(self, name:str, pipeline: TextClassificationPipeline, device: str):
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device
    
    def forward_func(self, inputs: tensor, position = 0):
        """
            Wrapper around prediction method of pipeline
        """
        pred = self.__pipeline.model(inputs,
                       attention_mask=torch.ones_like(inputs))
        return pred[position]
    

        
    def visualize(self, inputs: list, attributes: list, delta, outfile_path: str, prediction, target_label):
        """
            Visualization method.
            Takes list of inputs and correspondent attributs for them to visualize in a barplot
        """
        #import pdb; pdb.set_trace()
        vis_data_records = []
        attr_sum = attributes.sum(-1) 
        
        attr = attr_sum / torch.norm(attr_sum)
        html_content = show_text_attr(attr.cpu().numpy()[0], self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy()[0]))
        html = html_content.data
        with open(f"{outfile_path}.html", 'w') as f:
            f.write(html)
        """
        vis_data_records.append(visualization.VisualizationDataRecord(
                                attr.detach().cpu().numpy(),
                                prediction[0]["score"],
                                round(prediction[0]["score"]),
                                target_label,
                                target_label,
                                attr.sum().detach().cpu().numpy(),
                                self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy()[0])[::-1],
                                delta,
            )
        )
        html_content = visualization.visualize_text(vis_data_records)
        html = html_content.data
        with open(f"{outfile_path}", 'w') as f:
            f.write(html)
        """
        """
        a = pd.Series(attr.cpu().numpy()[0][::-1], 
                         index = self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy()[0])[::-1])
        
        a.plot.barh(figsize=(10,20))
        plt.savefig(outfile_path)"""


    def explain(self, text: str, outfile_path: str):
        """
            Main entry method. Passes text through series of transformations and through the model. 
            Calls visualization method.
        """
        prediction = self.__pipeline.predict(text)
        inputs = self.generate_inputs(text)
        baseline = self.generate_baseline(sequence_len = inputs.shape[1])
        
        lig = LayerIntegratedGradients(self.forward_func, getattr(self.__pipeline.model, 'deberta').embeddings)
        target_label = self.__pipeline.model.config.label2id[prediction[0]['label']]
        attributes, delta = lig.attribute(inputs=inputs,
                                  baselines=baseline,
                                  target = target_label, 
                                  return_convergence_delta = True)
        # Give a path to save
        self.visualize(inputs, attributes, delta, outfile_path, prediction, target_label)
    
    def generate_inputs(self, text: str) -> tensor:
        """
            Convenience method for generation of input ids as list of torch tensors
        """
        return torch.tensor(self.__pipeline.tokenizer.encode(text, add_special_tokens=False), device = self.__device).unsqueeze(0)
    
    def generate_baseline(self, sequence_len: int) -> tensor:
        """
            Convenience method for generation of baseline vector as list of torch tensors
        """        
        return torch.tensor([self.__pipeline.tokenizer.cls_token_id] + [self.__pipeline.tokenizer.pad_token_id] * (sequence_len - 2) + [self.__pipeline.tokenizer.sep_token_id], device = self.__device).unsqueeze(0)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint) 
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=args.num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf = transformers.pipeline("text-classification", 
                                model=model, 
                                tokenizer=tokenizer, 
                                device=device
                                )
    exp_model = ExplainableTransformerPipeline(args.model_checkpoint, clf, device)

    idx=0
    with jsonlines.open(args.a1_analysis_file, 'r') as reader:
        for obj in reader:
            exp_model.explain(obj["review"], os.path.join(args.output_dir,f'example_{idx}'))
            idx+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis_dir', default='/scratch/general/vast/u1419540/cs6966/assignment3/out/', type=str, help='Directory where attribution figures will be saved')
    parser.add_argument('--model_checkpoint', type=str, default='microsoft/deberta-v3-base', help='model checkpoint')
    parser.add_argument('--a1_analysis_file', type=str, default='out/a1_analysis_data.jsonl', help='path to a1 analysis file')
    parser.add_argument('--num_labels', default=2, type=int, help='Task number of labels')
    parser.add_argument('--output_dir', default='/scratch/general/vast/u1419540/cs6966/assignment3/out/', type=str, help='Directory where model checkpoints will be saved')    
    args = parser.parse_args()
    main(args)


