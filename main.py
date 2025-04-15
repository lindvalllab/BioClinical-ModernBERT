import argparse
import os
import torch
import time
from transformers import set_seed

from src.dataloader.dataloader import get_data
from src.tasks.classification import ClassificationTrainer

project_root = os.path.dirname(os.path.abspath(__file__))

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def main(args):
    # Set seed for reproducibility.
    set_seed(args.seed)
    
    device = get_device()
    print(f"Using device: {device}")

    data_wrapper = get_data(args.dataset)
    
    # Prepare training hyperparameters.
    training_args = {
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "weight_decay": args.wd,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation_steps
    }
    
    # Calculate effective batch size.
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"Effective batch size: {effective_batch_size}")
    
    # Use a common base string to avoid redundancy.
    base_name = (
        f"lr={args.lr}_wd={args.wd}_epochs={args.epochs}_"
        f"seed={args.seed}_effective_batch_size={effective_batch_size}"
    )
    
    # Create log file path.
    model_name = args.model.split("/")[-1]
    log_output_dir = os.path.join(project_root, "outputs", args.dataset, model_name)
    os.makedirs(log_output_dir, exist_ok=True)
    log_file = os.path.join(log_output_dir, base_name + ".txt")
    
    # Create checkpoint directory path.
    checkpoint_dir = os.path.join(project_root, "checkpoints", args.dataset, model_name, base_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {checkpoint_dir}")
    
    # Initialize the correct trainer based on the problem type.
    if "classification" in data_wrapper.problem_type:
        trainer_obj = ClassificationTrainer(
            device, args.model, data_wrapper, training_args, checkpoint_dir
        )
    else:
        raise NotImplementedError(f"The task {data_wrapper.problem_type} is not implemented.")
    
    # Time the training process.
    start_train = time.time()
    trainer_obj.train()
    end_train = time.time()
    train_duration = end_train - start_train
    print(f"Training took {train_duration:.2f} seconds.")
    
    # Time the evaluation process.
    start_eval = time.time()
    trainer_obj.evaluate()
    end_eval = time.time()
    eval_duration = end_eval - start_eval
    print(f"Evaluation took {eval_duration:.2f} seconds.")
    
    print("Test set evaluation:", trainer_obj.test_results)
    
    # Determine the best epoch from trainer's log history.
    best_epoch = None
    best_metric = None
    for log in trainer_obj.trainer.state.log_history:
        # Only look at evaluation logs (which include an 'epoch' key and your metric key).
        if "epoch" in log and "eval_weighted_f1" in log:
            # If no best_metric set yet, or this eval is better, update best_epoch.
            if best_metric is None or log["eval_weighted_f1"] > best_metric:
                best_metric = log["eval_weighted_f1"]
                best_epoch = log["epoch"]
    
    if best_epoch is not None:
        print(f"Best epoch: {best_epoch}")
    else:
        print("Best epoch could not be determined from the log history.")
    
    # Log output: save evaluation results, best epoch, and timing information.
    with open(log_file, "w") as f:
        f.write("Test evaluation results:\n")
        for key, value in trainer_obj.test_results.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nSeed: {args.seed}\n")
        f.write(f"Effective batch size: {effective_batch_size}\n")
        if best_epoch is not None:
            f.write(f"Best epoch selected: {best_epoch}\n")
        f.write(f"Training duration (seconds): {train_duration:.2f}\n")
        f.write(f"Evaluation duration (seconds): {eval_duration:.2f}\n")
    
    print(f"Test results and timings logged to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to evaluate encoders on downstream tasks."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset for downstream task (e.g. HOC)")
    parser.add_argument("--model", type=str, required=True, help="HF Model to evaluate (e.g. answerdotai/ModernBERT-base)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device for training and evaluation")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()
    main(args)