import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
import numpy as np

def evaluate_model(dataset_name, model_name, subset_size=5):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Prepare dataset inputs
    def prepare_inputs(examples):
        return tokenizer(
            examples["question"], 
            truncation=True, 
            padding="max_length", 
            max_length=512, 
            return_tensors="pt"
        )

    eval_dataset = dataset["train"].map(prepare_inputs, batched=True)
    eval_subset = eval_dataset.select(range(subset_size))

    # Generate predictions
    def generate_predictions(batch):
        input_ids = torch.tensor(batch['input_ids']).to(model.device)
        attention_mask = torch.tensor(batch['attention_mask']).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_length=700
            )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predictions = generate_predictions(eval_subset)
    references = dataset["train"]["answer"][:subset_size]

    # Compute ROUGE and BLEU metrics
    def compute_rouge(predictions, references):
        rouge_metric = evaluate.load("rouge")
        return rouge_metric.compute(predictions=predictions, references=references)

    def compute_bleu(predictions, references):
        bleu_metric = evaluate.load("bleu")
        return bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])

    # Compute classification metrics
    def compute_classification_metrics(predictions, references):
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        true_labels = [1 if pred == ref else 0 for pred, ref in zip(predictions, references)]
        
        precision = precision_metric.compute(predictions=true_labels, references=[1] * len(true_labels))
        recall = recall_metric.compute(predictions=true_labels, references=[1] * len(true_labels))
        accuracy = accuracy_metric.compute(predictions=true_labels, references=[1] * len(true_labels))
        f1 = f1_metric.compute(predictions=true_labels, references=[1] * len(true_labels))

        return {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1
        }

    # Compute perplexity
    def calculate_perplexity(probabilities):
        log_likelihood = -np.mean(np.log(probabilities.cpu().numpy()))
        return np.exp(log_likelihood)

    input_ids = tokenizer(references, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        perplexity = calculate_perplexity(probabilities)

    # Return metrics
    return {
        "rouge_scores": compute_rouge(predictions, references),
        "bleu_scores": compute_bleu(predictions, references),
        "classification_metrics": compute_classification_metrics(predictions, references),
        "perplexity": perplexity
    }
