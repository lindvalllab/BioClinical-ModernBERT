import numpy as np
import evaluate
import torch
from tqdm import tqdm
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


def predict_answer_mlm(example, model, tokenizer, option_mapping, option_tokens, model_name):
    """
    Given a formatted prompt example with a masked token, this function performs 
    a forward pass through the model, locates the [MASK] token, and computes 
    probabilities over candidate answer tokens.
    
    Args:
        example (dict): A dictionary with keys 'text' and 'label' (gold answer).
        model (PreTrainedModel): The masked language model.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        option_mapping (dict): Mapping from candidate token strings to canonical answer letters.
        option_tokens (list): List of candidate token strings to check (e.g., "A", "B", etc.).
        model_name (str): The model name (used for any necessary text adjustments).
    
    Returns:
        str: The predicted answer letter.
    """
    inp_text = example['text']
    
    # Tokenize the prompt text.
    inputs = tokenizer(inp_text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    
    # Locate all indices for the mask token; use the last occurrence.
    mask_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero()
    mask_logits = outputs.logits[0, mask_indices[-1, 1]]
    
    # Compute softmax probabilities over the vocabulary.
    probabilities = torch.nn.functional.softmax(mask_logits, dim=-1)
    
    # For each candidate option, get the probability of its token.
    option_ids = [tokenizer.encode(t)[1] for t in option_tokens]
    option_probs = {}
    for token, token_id in zip(option_tokens, option_ids):
        canonical = option_mapping[token]
        option_probs[canonical] = option_probs.get(canonical, 0) + probabilities[token_id].item()
    
    # Select the option (letter) with the highest probability.
    predicted_token = max(option_probs.items(), key=lambda x: x[1])[0]
    return predicted_token


def evaluate_mmlu(model, tokenizer, examples):
    """
    Evaluates the model on a list of formatted MMLU examples.
    
    It runs the prediction function on each example, compares the predicted 
    answer letter to the gold answer, and computes overall accuracy.
    
    Args:
        model (PreTrainedModel): The masked language model.
        tokenizer (PreTrainedTokenizer): The model's tokenizer.
        examples (list): List of formatted examples (each a dict with 'text' and 'label').
    
    Returns:
        tuple: (accuracy, predictions list, gold answers list)
    """

    # Deprecated but kept for legacy reasons pre-tokenizer fix.
    option_mapping = {
        "A": "A", " A": "A",
        "B": "B", " B": "B",
        "C": "C", " C": "C",
        "D": "D", " D": "D",
    }
    option_tokens = ["A", " A", "B", " B",
                     "C", " C", "D", " D",]
    
    preds = []
    golds = []
    
    # Evaluate each example.
    for ex in tqdm(examples, desc="Evaluating MMLU"):
        gold = ex.get('label')
        pred = predict_answer_mlm(ex, model, tokenizer, option_mapping, option_tokens, model.config._name_or_path)
        preds.append(pred.strip())
        golds.append(gold.strip())
    
    # Compute accuracy over the entire dataset.
    acc = accuracy_score(golds, preds)
    return {"accuracy": acc}

def compute_metrics_mlm_for_training(eval_pred):
    """
    eval_pred.predictions: np.ndarray of shape (batch, seq_len, vocab_size)
    eval_pred.label_ids:      np.ndarray of shape (batch, seq_len)
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    # get the predicted token‐ids at each position
    pred_ids = logits.argmax(axis=-1)

    # only keep the positions where label != -100 (i.e., your masked token)
    mask = labels != -100

    # flatten and compare
    correct = (pred_ids[mask] == labels[mask]).astype(np.float32)
    acc     = correct.mean().item()
    return {"accuracy": acc}