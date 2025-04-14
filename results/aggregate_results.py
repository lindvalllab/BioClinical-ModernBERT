import os
import glob
import re
import argparse
import pandas as pd

def parse_filename(filename):
    """
    Extract hyperparameters from the filename.
    Expected format: lr=<lr>_wd=<wd>_epochs=<epochs>_seed=<seed>_effective_batch_size=<effective_bs>.txt
    """
    pattern = r"lr=([^_]+)_wd=([^_]+)_epochs=([^_]+)_seed=([^_]+)_effective_batch_size=([^\.]+)"
    match = re.search(pattern, filename)
    if match:
        lr, wd, epochs, seed, effective_bs = match.groups()
        return {
            "lr": float(lr),
            "wd": float(wd),
            "epochs": int(epochs),
            "seed": int(seed),
            "effective_batch_size": int(effective_bs)
        }
    else:
        return None

def parse_log_file(filepath):
    """
    Parse the log file content, which is assumed to contain lines of the form 'Key: Value'.
    Returns a dictionary of keys and values.
    """
    data = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip headers like "Test evaluation results:" that do not contain ':'
        if ':' not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        # Try to convert to float; if an integer is desired but a float conversion succeeds, that's fine.
        try:
            if "." in value:
                data[key] = float(value)
            else:
                data[key] = int(value)
        except ValueError:
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = value  # if conversion fails, keep as string
    return data

def main(results_path):
    records = []
    
    # Recursively find all .txt files in the directory.
    filepaths = glob.glob(os.path.join(results_path, "**", "*.txt"), recursive=True)
    if not filepaths:
        print(f"No log files found in {results_path}")
        return

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        # Assume that the model name is the immediate parent folder
        model = os.path.basename(os.path.dirname(filepath))
        
        params = parse_filename(filename)
        if params is None:
            print(f"Filename {filename} does not match the expected pattern. Skipping file.")
            continue
        
        log_data = parse_log_file(filepath)
        
        # Combine the data from the filename, file content, and model name.
        record = {
            "Model": model,
            **params,  # lr, wd, epochs, seed, effective_batch_size
        }
        # Merge the metrics from log_data.
        record.update(log_data)
        records.append(record)
    
    # Create a DataFrame from the list of dictionaries.
    df = pd.DataFrame(records)
    print("Individual results:")
    print(df.head(), "\n")
    
    # We group on Model, lr, wd, and effective_batch_size.
    group_cols = ["Model", "lr", "wd", "effective_batch_size"]
    
    # Identify metric columns: ignore hyperparameter columns and seed/epochs.
    ignore_cols = set(group_cols + ["seed", "epochs"])
    metric_cols = [col for col in df.columns if col not in ignore_cols]
    
    # Define aggregation: for each metric, compute mean, median, min, and max.
    agg_funcs = {col: ["mean", "median", "min", "max"] for col in metric_cols}
    grouped = df.groupby(group_cols).agg(agg_funcs)
    
    # Flatten the multi-level column index.
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    print("Aggregated results over seeds:")
    print(grouped.head())
    
    # Optionally, save the aggregated DataFrame to CSV.
    output_csv = os.path.join(results_path, "aggregated_results.csv")
    grouped.to_csv(output_csv, index=False)
    print(f"\nAggregated results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate experiment log results over seeds.")
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to the results directory (e.g., results/dataset)")
    args = parser.parse_args()
    main(args.results_path)