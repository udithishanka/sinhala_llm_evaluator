import matplotlib.pyplot as plt
import numpy as np

from llm_evaluator.eval import evaluate_model

def compare_models(
    dataset_name: str,
    model1_name: str,
    model2_name: str,
    max_length: int = 2048,
    subset_size: int = 5,
    output_plot: str = "model_comparison.png"
):
    """
    Compare two language models on a given dataset using various metrics
    (METEOR, ROUGE, BLEU, Perplexity) and display the results side by side,
    along with a bar chart.

    Args:
        dataset_name (str): Name/path of the Hugging Face dataset
        model1_name (str): Name or path of the first model
        model2_name (str): Name or path of the second model
        max_length (int, optional): Max input length for tokenization
        subset_size (int, optional): Number of samples to evaluate
        output_plot (str, optional): File name for saving the bar chart

    Returns:
        dict: A dictionary with metrics for both models:
        {
            "model1": { "meteor": ..., "rouge1": ..., ...},
            "model2": { "meteor": ..., "rouge1": ..., ...}
        }
    """

    print(f"Evaluating Model 1: {model1_name}")
    results1 = evaluate_model(
        dataset_name=dataset_name,
        model_name=model1_name,
        max_length=max_length,
        subset_size=subset_size
    )

    print(f"\nEvaluating Model 2: {model2_name}")
    results2 = evaluate_model(
        dataset_name=dataset_name,
        model_name=model2_name,
        max_length=max_length,
        subset_size=subset_size
    )

    # Gather metrics in a consistent structure
    # ------------------------------------------------
    # 'results1' and 'results2' look like:
    # {
    #     "meteor_scores": {"meteor": float},
    #     "rouge_scores": {"rouge1": float, "rouge2": float, "rougeL": float, "rougeLsum": float},
    #     "bleu_scores": {"bleu": float, ...},
    #     "perplexity": float
    # }

    model1_metrics = {
        "meteor": results1["meteor_scores"]["meteor"],
        "rouge1": results1["rouge_scores"]["rouge1"],
        "rouge2": results1["rouge_scores"]["rouge2"],
        "rougeL": results1["rouge_scores"]["rougeL"],
        "rougeLsum": results1["rouge_scores"]["rougeLsum"],
        "bleu": results1["bleu_scores"]["bleu"],
        "perplexity": results1["perplexity"],
    }

    model2_metrics = {
        "meteor": results2["meteor_scores"]["meteor"],
        "rouge1": results2["rouge_scores"]["rouge1"],
        "rouge2": results2["rouge_scores"]["rouge2"],
        "rougeL": results2["rouge_scores"]["rougeL"],
        "rougeLsum": results2["rouge_scores"]["rougeLsum"],
        "bleu": results2["bleu_scores"]["bleu"],
        "perplexity": results2["perplexity"],
    }

    # Print results side by side
    # ------------------------------------------------
    print("\nComparison of Metrics\n" + "-"*50)
    metric_names = list(model1_metrics.keys())  # or define your own order
    for metric in metric_names:
        val1 = model1_metrics[metric]
        val2 = model2_metrics[metric]
        print(f"{metric.upper()}:\n  Model1 = {val1:.4f}\n  Model2 = {val2:.4f}\n")

    # Create bar chart
    # ------------------------------------------------
    x = np.arange(len(metric_names))
    bar_width = 0.35

    model1_values = [model1_metrics[m] for m in metric_names]
    model2_values = [model2_metrics[m] for m in metric_names]

    fig, ax = plt.subplots(figsize=(8, 5))

    rects1 = ax.bar(x - bar_width/2, model1_values, bar_width, label=model1_name, alpha=0.7)
    rects2 = ax.bar(x + bar_width/2, model2_values, bar_width, label=model2_name, alpha=0.7)

    ax.set_ylabel("Scores")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.legend()

    # Annotate bars (optional)    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8
            )

    plt.tight_layout()
    plt.savefig(output_plot)
    plt.show()

    return {
        "model1": model1_metrics,
        "model2": model2_metrics
    }
