import numpy as np
import evaluate
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics_multi_label_classification(eval_pred):
    """
    Compute evaluation metrics for multi-label classification.
    For multi-label tasks we use a threshold of 0 (since the model outputs logits).
    This function computes both weighted and micro F1 scores.
    You can extend this function or choose different strategies depending on the dataset.
    """
    logits, labels = eval_pred
    # Convert logits to binary predictions using a threshold of 0.
    preds = (logits > 0).astype(int)
    
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    
    return {"weighted_f1": weighted_f1, "micro_f1": micro_f1}


def compute_metrics_single_label_classification(eval_pred):
    """
    Compute evaluation metrics for single-label classification.
    
    Given eval_pred as a tuple (logits, labels) where:
       - logits: a numpy array of shape (batch_size, num_labels)
       - labels: a numpy array of shape (batch_size)
       
    This function computes predictions by taking the argmax of the logits along the labels axis.
    It then calculates accuracy and F1 scores (both weighted and micro) using scikit‑learn.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    return {"accuracy": accuracy, "weighted_f1": weighted_f1, "micro_f1": micro_f1}


def compute_metrics_token_classification(eval_pred, id2label):
    """
    Compute evaluation metrics for single-label classification.
    
    Given eval_pred as a tuple (logits, labels) where:
       - logits: a numpy array of shape (batch_size, num_labels)
       - labels: a numpy array of shape (batch_size)
       
    This function computes predictions by taking the argmax of the logits along the labels axis.
    It then calculates accuracy and F1 scores (both weighted and micro) using scikit‑learn.
    """
    seqeval = evaluate.load("seqeval")
    predictions, labels = eval_pred
    # Get the most probable label for each token.
    predictions = np.argmax(predictions, axis=2)
    true_predictions = []
    true_labels = []
    for pred, label in zip(predictions, labels):
        pred_labels = []
        gold_labels = []
        for p, l in zip(pred, label):
            if l != -100:
                pred_labels.append(id2label[p])
                gold_labels.append(id2label[l])
        true_predictions.append(pred_labels)
        true_labels.append(gold_labels)
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }