import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
import numpy as np
import time
import gc

def evaluate_model(dataset_name, model_name, max_length=2048, subset_size=5):
    """_summary_

    Args:
        dataset_name (_type_): _description_
        model_name (_type_): _description_
        max_length (int, optional): _description_. Defaults to 2048.
        subset_size (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    start_time = time.time()
    
    if isinstance(model_name, str):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = model_name
        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
        
    dataset = load_dataset(dataset_name)
    # Prepare dataset inputs
    def prepare_inputs(examples):
        return tokenizer(
            examples["question"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length, 
            return_tensors="pt"
        )

    eval_dataset = dataset["train"].map(prepare_inputs, batched=True)
    eval_subset = eval_dataset.select(range(subset_size))
    
    # Generate predictions
    def generate_predictions(batch):
        print("Generating predictions...")
        input_ids = torch.tensor(batch['input_ids']).to(model.device)
        attention_mask = torch.tensor(batch['attention_mask']).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=200,
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
    # def compute_classification_metrics(predictions, references):
    #     precision_metric = evaluate.load("precision")
    #     recall_metric = evaluate.load("recall")
    #     accuracy_metric = evaluate.load("accuracy")
    #     f1_metric = evaluate.load("f1")

    #     precision = precision_metric.compute(predictions=predictions, references=references)
    #     recall = recall_metric.compute(predictions=predictions, references=references)
    #     accuracy = accuracy_metric.compute(predictions=predictions, references=references)
    #     f1 = f1_metric.compute(predictions=predictions, references=references)
        
    #     return {
    #         "precision": precision['precision'],
    #         "recall": recall['recall'],
    #         "accuracy": accuracy['accuracy'],
    #         "f1": f1['f1']
    #     }
    
    def compute_meteor(predictions, references):
        """
        Computes the METEOR (Metric for Evaluation of Translation with Explicit Ordering) score.
        Assumes predictions and references are strings (sequence-level).
        """
        meteor_metric = evaluate.load("meteor")
        meteor = meteor_metric.compute(predictions=predictions, references=references)
        return {"meteor": meteor['meteor']}

    # Compute perplexity
    def calculate_perplexity(probabilities):
        log_likelihood = -np.mean(np.log(probabilities.cpu().numpy()))
        return np.exp(log_likelihood)

    print("Evaluating...")
    input_ids = tokenizer(references, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        perplexity = calculate_perplexity(probabilities)

    meteor_scores = compute_meteor(predictions, references)
    rouge_scores = compute_rouge(predictions, references)
    bleu_scores = compute_bleu(predictions, references)
    
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60
    print(f"Time taken for evaluation: {elapsed_minutes:.2f} minutes")

    del predictions
    del input_ids
    del outputs
    del logits
    del probabilities
    del references
    del eval_dataset
    del eval_subset
    del dataset
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        # "classification_metrics": compute_classification_metrics(predictions, references),
        "meteor_scores": meteor_scores,
        "rouge_scores": rouge_scores,
        "bleu_scores": bleu_scores,
        "perplexity": perplexity
    }
