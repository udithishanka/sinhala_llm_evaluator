import torch
import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

class LLMEvaluator:
    def __init__(self, model_name, subset_size=5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.subset_size = subset_size
        
        # Load evaluation metrics
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        self.f1_metric = evaluate.load("f1")
        self.accuracy_metric = evaluate.load("accuracy")
        self.precision_metric = evaluate.load("precision")
        self.recall_metric = evaluate.load("recall")

    def prepare_inputs(self, examples):
        inputs = self.tokenizer(examples["question"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        return inputs
    
    def select_subset(self, dataset, split="train"):
        eval_dataset = dataset[split].map(self.prepare_inputs, batched=True)
        eval_subset = eval_dataset.select(range(self.subset_size))
        return eval_subset
    
    def generate_predictions(self, batch):
        input_ids = torch.tensor(batch['input_ids']).to(self.model.device)
        attention_mask = torch.tensor(batch['attention_mask']).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=700)
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def compute_rouge(self, predictions, references):
        return self.rouge_metric.compute(predictions=predictions, references=references)

    def compute_bleu(self, predictions, references):
        return self.bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])

    def compute_classification_metrics(self, predictions, references):
        true_labels = [1 if pred == ref else 0 for pred, ref in zip(predictions, references)]
        
        precision = self.precision_metric.compute(predictions=true_labels, references=[1] * len(true_labels))
        recall = self.recall_metric.compute(predictions=true_labels, references=[1] * len(true_labels))
        accuracy = self.accuracy_metric.compute(predictions=true_labels, references=[1] * len(true_labels))
        f1 = self.f1_metric.compute(predictions=true_labels, references=[1] * len(true_labels))

        return {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1
        }

    def calculate_perplexity(self, predictions):
        log_likelihood = -np.mean(np.log(predictions.cpu().numpy()))
        perplexity = np.exp(log_likelihood)
        return perplexity

    def compute_perplexity(self, references):
        input_ids = self.tokenizer(references, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        perplexity = self.calculate_perplexity(probabilities)
        return perplexity

    def evaluate(self, dataset, split="train"):
        eval_subset = self.select_subset(dataset, split)
        predictions = self.generate_predictions(eval_subset)
        references = dataset[split]["answer"][:self.subset_size]
        
        rouge_scores = self.compute_rouge(predictions, references)
        bleu_scores = self.compute_bleu(predictions, references)
        classification_metrics = self.compute_classification_metrics(predictions, references)
        perplexity = self.compute_perplexity(references)

        return {
            "ROUGE": rouge_scores,
            "BLEU": bleu_scores,
            "Classification Metrics": classification_metrics,
            "Perplexity": perplexity
        }
