## MODIFIED FROM https://github.com/AnswerDotAI/ModernBERT/blob/8c57a0f01c12c4953ead53d398a36f81a4ba9e38/efficiency/multiprocess_bench.py

import copy
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import transformers
import torch
import random
import numpy as np
import time
from transformers import AutoModel, AutoTokenizer
import srsly
import os
import gc
from multiprocessing import Process, Queue

def create_fixed_short_dataset(tokenizer, num_samples=8192):
    tokens = torch.randint(100, 16000, (num_samples, 512))
    mask = torch.ones(num_samples, 512)
    return {
        'input_ids': tokens.long(),
        'attention_mask': mask.float()
    }

def create_fixed_medium_dataset(tokenizer, num_samples=8192):
    tokens = torch.randint(100, 16000, (num_samples, 4096))
    mask = torch.ones(num_samples, 4096)
    return {
        'input_ids': tokens.long(),
        'attention_mask': mask.float()
    }


def create_fixed_long_dataset(tokenizer, num_samples=8192):
    tokens = torch.randint(100, 16000, (num_samples, 8192))
    mask = torch.ones(num_samples, 8192)
    return {
        'input_ids': tokens.long(),
        'attention_mask': mask.float()
    }

def create_variable_short_dataset(tokenizer, num_samples=8192):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    lengths = torch.normal(mean=256, std=64, size=(num_samples,)).int().clamp(16, 512)
    tokens_list = []
    masks_list = []
    for length in lengths:
        tokens = torch.randint(100, 16000, (length.item(),))
        mask = torch.ones(length.item())
        padded_tokens = torch.full((512,), tokenizer.pad_token_id, dtype=torch.long)
        padded_mask = torch.zeros(512)
        padded_tokens[:length] = tokens
        padded_mask[:length] = mask
        tokens_list.append(padded_tokens)
        masks_list.append(padded_mask)
    
    return {
        'input_ids': torch.stack(tokens_list),
        'attention_mask': torch.stack(masks_list)
    }

def create_variable_medium_dataset(tokenizer, num_samples=8192):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    lengths = torch.normal(mean=2048, std=512, size=(num_samples,)).int().clamp(16, 4096)
    tokens_list = []
    masks_list = []
    for length in lengths:
        tokens = torch.randint(100, 16000, (length.item(),))
        mask = torch.ones(length.item())
        padded_tokens = torch.full((4096,), tokenizer.pad_token_id, dtype=torch.long)
        padded_mask = torch.zeros(4096)
        padded_tokens[:length] = tokens
        padded_mask[:length] = mask
        tokens_list.append(padded_tokens)
        masks_list.append(padded_mask)
    
    return {
        'input_ids': torch.stack(tokens_list),
        'attention_mask': torch.stack(masks_list)
    }

def create_variable_long_dataset(tokenizer, num_samples=8192):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    lengths = torch.normal(mean=4096, std=1024, size=(num_samples,)).int().clamp(16, 8192)
    tokens_list = []
    masks_list = []
    for length in lengths:
        tokens = torch.randint(100, 16000, (length.item(),))
        mask = torch.ones(length.item())
        padded_tokens = torch.full((8192,), tokenizer.pad_token_id, dtype=torch.long)
        padded_mask = torch.zeros(8192)
        padded_tokens[:length] = tokens
        padded_mask[:length] = mask
        tokens_list.append(padded_tokens)
        masks_list.append(padded_mask)
    
    return {
        'input_ids': torch.stack(tokens_list),
        'attention_mask': torch.stack(masks_list)
    }

def create_all_datasets(tokenizer, num_samples=8192):
    max_length = tokenizer.model_max_length if tokenizer.model_max_length < 10000 else 512
    dict_datasets = {
        'fixed_short': create_fixed_short_dataset(tokenizer, num_samples),
        'variable_short': create_variable_short_dataset(tokenizer, num_samples),
    }
    if max_length > 512:
        dict_datasets['fixed_medium'] = create_fixed_medium_dataset(tokenizer, num_samples)
        dict_datasets['variable_medium'] = create_variable_medium_dataset(tokenizer, num_samples)
    if max_length > 4096:
        dict_datasets['fixed_long'] = create_fixed_long_dataset(tokenizer, num_samples)
        dict_datasets['variable_long'] = create_variable_long_dataset(tokenizer, num_samples)
    return dict_datasets

def test_batch_size_worker(q, model_name, input_ids, attention_mask, bsize, device, use_xformers):
    """
    Worker that:
    1. Loads the model
    2. Tries given batch size
    3. Returns success or fail
    """
    try:
        if 'gte' in model_name.lower() and use_xformers:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=False
            )
            model.config.use_memory_efficient_attention = True
        else:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=False
            )
        model = model.to(device)

        with torch.no_grad():
            for i in range(0, input_ids.size(0), bsize):
                batch_ids  = input_ids[i:i+bsize].to(device)
                batch_mask = attention_mask[i:i+bsize].to(device)
                model(input_ids=batch_ids, attention_mask=batch_mask)
        q.put(('success', True))
    except RuntimeError:
        q.put(('success', False))
    except Exception as e:
        q.put(('error', str(e)))

def find_max_batch_size_worker(q, model_name, input_ids, attention_mask, device, use_xformers):
    """
    Worker that runs the batch size finding logic.
    Each attempt is run in its own worker to ensure full memory isolation.
    """

    def try_batch_size(bsize):
        print(f"Attempting batch size: {bsize}")
        # Spawn a worker for each attempt
        attempt_q = Queue()
        p = Process(
            target=test_batch_size_worker,
            args=(attempt_q, model_name, input_ids, attention_mask, bsize, device, use_xformers)
        )
        p.start()
        p.join()
        result = attempt_q.get()
        p = None
        if result[0] == 'error':
            # If there's an error unrelated to OOM, raise it
            print(f"Error occurred: {result[1]}")
            raise RuntimeError(result[1])
        success = result[1]
        print(f"Batch size {bsize}: {'succeeded' if success else 'failed'}")

        print("Clearing CUDA cache and garbage collection")
        torch.cuda.empty_cache()
        gc.collect()
        return success

    try:
        print("\nStarting batch size search...")
        batch_size = 2740
        print("\nPhase 1: Increasing batch size until OOM")
        # Increase by 16 until OOM or max 4096
        while try_batch_size(batch_size) and batch_size < 4096:
            batch_size += 16
            print(f"Increasing to {batch_size}")

        print("\nPhase 2: Backing off by 32 until stable")
        # Back off by 32 until stable
        while not try_batch_size(batch_size) and batch_size > 64:
            batch_size -= 64
            print(f"Decreasing to {batch_size}")

        # If still not working, try smaller decrements
        if not try_batch_size(batch_size):
            print("\nPhase 3: Fine-tuning with smaller decrements")
            while not try_batch_size(batch_size) and batch_size > 4:
                batch_size -= 4
                print(f"Fine-tuning decrease to {batch_size}")
            if batch_size <= 4 and not try_batch_size(batch_size):
                print("Attempting minimum batch size of 1")
                batch_size = 1
                if not try_batch_size(batch_size):
                    raise RuntimeError("Cannot find a working batch size.")

        print("\nPhase 4: Final optimization")
        
        # Try increments of 32
        test_size = batch_size + 32
        while test_size < 4096:
            success = try_batch_size(test_size)
            if not success:
                test_size = batch_size
                break
            batch_size = test_size
            test_size += 32
            print(f"Testing increment to {test_size}")
            
        # Try increments of 16
        test_size = batch_size + 16
        while test_size < 4096:
            success = try_batch_size(test_size)
            if not success:
                test_size = batch_size
                break
            batch_size = test_size
            test_size += 16
            print(f"Testing increment to {test_size}")
            
        # Try increments of 8
        test_size = batch_size + 8
        while test_size < 4096:
            success = try_batch_size(test_size)
            if not success:
                test_size = batch_size
                break
            batch_size = test_size
            test_size += 8
            print(f"Testing increment to {test_size}")
            
        # Try increments of 4
        test_size = batch_size + 4
        while test_size < 4096:
            success = try_batch_size(test_size)
            if not success:
                test_size = batch_size
                break
            batch_size = test_size
            test_size += 4
            print(f"Testing increment to {test_size}")
            
        # Try increments of 2
        test_size = batch_size + 2
        while test_size < 4096:
            success = try_batch_size(test_size)
            if not success:
                test_size = batch_size
                break
            batch_size = test_size
            test_size += 2
            print(f"Testing increment to {test_size}")

        final_batch_size = min(batch_size, 4096)
        if final_batch_size > 8:
            final_batch_size -= 4
        print(f"\nFinal batch size determined: {final_batch_size}")
        q.put(('success', final_batch_size))
    except Exception as e:
        print(f"Error in batch size search: {str(e)}")
        q.put(('error', str(e)))


def inference_worker(q, model_name, dataset_name, input_ids, attention_mask, max_batch_size, n_iters, device, use_xformers):
    """
    Worker to run inference multiple times and report mean/std of times.
    Model loading is done here to isolate memory usage.
    """
    try:
        if 'gte' in model_name.lower() and use_xformers:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=False
            )
            model.config.use_memory_efficient_attention = True
        else:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=False
            )
        model = model.to(device)
        model.eval()

        times = []
        for _ in tqdm(range(n_iters), desc=f"Inference [{dataset_name}]", position=0, leave=True):
            start_time = time.time()
            with torch.no_grad():
                for i in range(0, len(input_ids), max_batch_size):
                    batch_ids = input_ids[i:i+max_batch_size].clone().to(device)
                    batch_mask = attention_mask[i:i+max_batch_size].clone().to(device)
                    model(input_ids=batch_ids, attention_mask=batch_mask)
            end_time = time.time()
            times.append(end_time - start_time)

        mean_time = np.mean(times)
        std_time = np.std(times)
        q.put((dataset_name, mean_time, std_time, max_batch_size))
    except Exception as e:
        q.put(('error', str(e)))


def run_inference_benchmark(model_name, use_xformers=False, n_iters=10, gpu=0):
    device = f'cuda'
    num_samples = 4096
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
    datasets = create_all_datasets(tokenizer, num_samples)

    processing_times = {}
    fixed_batch_sizes = {}

    # Ensure a clean GPU state before starting
    torch.cuda.empty_cache()
    gc.collect()

    for dataset_name, dataset in datasets.items():
        input_ids = dataset['input_ids']
        attention_mask = dataset['attention_mask'].int()

        # Run batch size finding in its own worker
        q = Queue()
        p = Process(
            target=find_max_batch_size_worker,
            args=(q, model_name, input_ids, attention_mask, device, use_xformers)
        )
        p.start()
        p.join()
        result = q.get()
        p = None
        if result[0] == 'error':
            print(f"Error finding batch size for {dataset_name}: {result[1]}")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        max_batch_size = result[1]
        fixed_batch_sizes[dataset_name] = max_batch_size  # Save for reference

        torch.cuda.empty_cache()
        gc.collect()

        # Run inference in its own worker
        q = Queue()
        p = Process(
            target=inference_worker,
            args=(q, model_name, dataset_name, input_ids, attention_mask, max_batch_size, n_iters, device, use_xformers)
        )
        p.start()
        p.join()
        result = q.get()
        p = None
        if result[0] == 'error':
            print(f"Error during inference for {dataset_name}: {result[1]}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

        dataset_name_ret, mean_time, std_time, bsize = result
        processing_times[dataset_name_ret] = {
            'mean': mean_time,
            'std': std_time,
            'max_batch_size': bsize
        }
        print(f"{dataset_name_ret} -> {mean_time:.2f} ± {std_time:.2f} sec (batch_size: {bsize})")

        torch.cuda.empty_cache()
        gc.collect()


    print("\nProcessing Time Summary:")
    print("-" * 50)
    print(f"\n{model_name} Model:")
    for dataset_name, metrics in processing_times.items():
        print(f"{dataset_name}: {metrics['mean']:.2f} ± {metrics['std']:.2f} seconds (batch_size: {metrics['max_batch_size']})")

    print("\nInference Speed Summary (in thousands of tokens/sec):")
    print("-" * 50)

    sequence_lengths = {
        'fixed_short': 512,
        'variable_short': 256,
        'fixed_medium': 4096,
        'variable_medium': 2048,
        'fixed_long': 8192,
        'variable_long': 4096
    }

    for dataset_name, metrics in processing_times.items():
        seq_len = sequence_lengths.get(dataset_name)
        if seq_len is None:
            continue
        total_tokens = num_samples * seq_len
        k_tok_per_sec = total_tokens / 1000 / metrics['mean']
        metrics['ktok_per_sec'] = round(k_tok_per_sec, 2)
        print(f"{dataset_name}: {k_tok_per_sec:.2f} kTok/s (batch_size: {metrics['max_batch_size']})")



    try:
        if use_xformers:
            os.makedirs(f"results/{model_name}_xformers", exist_ok=True)
            srsly.write_json(f"results/{model_name}_xformers_inference_times.json", processing_times)
        else:
            os.makedirs(f"results/{model_name}", exist_ok=True)
            srsly.write_json(f"results/{model_name}_inference_times.json", processing_times)
    except Exception as e:
        print(f"Error saving results: {e}")

    return processing_times


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference benchmark')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number to use')
    parser.add_argument('--model', type=str, default="GTE", help='Model name to benchmark')
    parser.add_argument('--xformers', action='store_true', help='Use XFormers')
    
    args = parser.parse_args()
    processing_times = run_inference_benchmark(model_name=args.model, use_xformers=args.xformers, gpu=args.gpu)
