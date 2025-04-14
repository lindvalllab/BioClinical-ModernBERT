import numpy as np
from sklearn.metrics import f1_score

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