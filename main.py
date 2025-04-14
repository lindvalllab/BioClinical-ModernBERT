import argparse
import os
import torch
import time
from transformers import set_seed

from src.dataloader.dataloader import get_data
from src.tasks.classification import ClassificationTrainer

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def main(args):
    # Set the seed before proceeding.
    set_seed(args.seed)
    
    device = get_device()
    print(f"Using device: {device}")

    data_wrapper = get_data(args.dataset)

    # Prepare hyperparameters in a dictionary.
    training_args = {
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "weight_decay": args.wd,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation_steps
    }

    # Calculate the effective batch size
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"Effective batch size: {effective_batch_size}")

    if "classification" in data_wrapper.problem_type:
        trainer_obj = ClassificationTrainer(device, args.model, data_wrapper, training_args)
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

    # Create directory for saving test output if necessary.
    model_name = args.model.split("/")[-1]
    output_dir = f"outputs/{args.dataset}/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Log output file name (include learning rate, weight decay, effective batch size, etc.)
    output_file = os.path.join(
        output_dir,
        f"lr={args.lr}_wd={args.wd}_epochs={args.epochs}_seed={args.seed}_effective_batch_size={effective_batch_size}.txt"
    )

    # Write the results and timing to the log file.
    with open(output_file, "w") as f:
        f.write("Test evaluation results:\n")
        for key, value in trainer_obj.test_results.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nSeed: {args.seed}\n")
        f.write(f"Effective batch size: {effective_batch_size}\n")
        f.write(f"Training duration (seconds): {train_duration:.2f}\n")
        f.write(f"Evaluation duration (seconds): {eval_duration:.2f}\n")

    print(f"Test results and timings logged to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to evaluate encoders on downstream tasks.")
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