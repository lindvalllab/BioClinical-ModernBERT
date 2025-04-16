import numpy as np
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