# Inspired by MosaicML streaming library examples

# This script converts the text contained in a column "notes" of a csv to an MDS dataset with optional tokenizing. Feel free to adapt to your needs!
# This script also counts the number of tokens generated and appends it to a file if you want to keep track of it.

# This script supports large datasets by using streaming, without which your RAM would have to be big enough to fit the entire dataset.
#   However, this also means that the number of rows in the csv cannot be accessed by the script, so if you want a progress bar you have
#   to pass the row count separately.

# Usage:
#   --dataset:       Path to the input CSV file containing raw text data (required).
#   --row_count:     (Optional) Total number of rows in the dataset. Used to display a progress bar with known length.
#                    If not provided, the script will still run, but the progress bar will be indeterminate.
#   --out_dir:       Output directory where the MDS-formatted data will be written.
#                    A subdirectory will be created based on the dataset filename.
#   --out_token_counts: Path to a text file where the total number of tokens will be appended.
#   --compression:   (Optional) Compression method to use for MDSWriter (e.g., zstd, lz4, etc.).
#   --concat_tokens: Number of tokens to concatenate into a single sequence (default: 8192).
#   --tokenizer:     (Optional) HuggingFace tokenizer name or path (default: "answerdotai/ModernBERT-base").
#   --bos_text:      (Optional) Beginning-of-sequence token string (default: "[CLS]").
#   --eos_text:      (Optional) End-of-sequence token string (default: "[SEP]").
#   --no_wrap:       (Optional) If provided, disables BOS/EOS wrapping around each chunk.

# python convert_dataset_to_mds.py \
#  --dataset path/to/dataset.csv \
#  --row_count 42 \
#  --out_token_counts path/to/file/logging/token/counts.txt \
#  --out_dir path/to/out/dir

import os
import warnings
from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable

import datasets as hf_datasets
import numpy as np
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase




def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(description="Convert dataset into MDS format, optionally concatenating and tokenizing")
    parser.add_argument("--dataset", type=str, required=True, help="path to data as csv")
    parser.add_argument("--out_token_counts", type=str, required=True, help="path to save txt file with token counts")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--row_count", type=int, required=False, help="Number of rows in the dataset")
    parser.add_argument("--compression", type=str, default=None)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--concat_tokens", type=int, help="Convert text to tokens and concatenate up to this many tokens", default=8192
    )

    parser.add_argument("--tokenizer", type=str, required=False, default="answerdotai/ModernBERT-base")
    parser.add_argument("--bos_text", type=str, required=False, default="[CLS]")
    parser.add_argument("--eos_text", type=str, required=False, default="[SEP]")
    parser.add_argument("--no_wrap", default=False, action="store_true")

    parsed = parser.parse_args()

    if os.path.isdir(os.path.join(parsed.out_dir, os.path.basename(parsed.dataset).rsplit("_", 1)[0])):
        raise ValueError(
            f"--out_dir={parsed.out_dir} is not empty."
        )

    # Make sure we have needed concat options
    if parsed.concat_tokens is not None and isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None:
        parser.error("When setting --concat_tokens, you must specify a --tokenizer")

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ""
    if parsed.eos_text is None:
        parsed.eos_text = ""
    return parsed


class ConcatTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}
    """

    def __init__(
        self,
        csv_file: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
    ):
        self.tokenizer = tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap
        self.csv_file = csv_file
        self.token_counts = 0
        self.ds = hf_datasets.load_dataset("csv", data_files=self.csv_file, split="train", streaming=True)

        self.bos_tokens = self.tokenizer(self.bos_text, truncation=False, padding=False, add_special_tokens=False)[
            "input_ids"
        ]
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f"You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error."
            )

        self.eos_tokens = self.tokenizer(self.eos_text, truncation=False, padding=False, add_special_tokens=False)[
            "input_ids"
        ]
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f"You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error."
            )
        self.space_token = self.tokenizer(" ", truncation=False, padding=False, add_special_tokens=False)[
            "input_ids"
        ]

        eos_text_provided = self.eos_text != ""
        bos_text_provided = self.bos_text != ""
        test_text = self.tokenizer("", truncation=False, padding=False, add_special_tokens=False)
        if len(test_text["input_ids"]) > 0 and (eos_text_provided or bos_text_provided):
            print(test_text["input_ids"])
            message = (
                "both eos and bos"
                if eos_text_provided and bos_text_provided
                else ("eos_text" if eos_text_provided else "bos_text")
            )
            warnings.warn(
                f"The provided tokenizer adds special tokens, but you also specified {message}. This may result "
                "in duplicated special tokens. Please be sure this is what you intend."
            )

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        # Calculate the number of content tokens allowed between bos and eos.
        content_length = self.max_length - len(self.bos_tokens) - len(self.eos_tokens)
        # Process each sample individually.
        n_tokens = 0
        for sample in self.ds:
            if not sample["notes"]:
                continue
            encoded = self.tokenizer(sample["notes"], truncation=False, padding=False, add_special_tokens=False)
            iids = encoded["input_ids"]
            # Create a fresh buffer for this sample.
            sample_buffer = list(iids)
            # Yield as many complete segments as possible from this sample.
            while len(sample_buffer) >= content_length:
                chunk = sample_buffer[:content_length]
                sample_buffer = sample_buffer[content_length:]
                first_token_with_space = self.tokenizer.encode(" " + self.tokenizer.decode(chunk[0]), add_special_tokens=False)
                tokens = self.bos_tokens + first_token_with_space + chunk[1:] + self.eos_tokens
                n_tokens += len(tokens)
                yield {"input_ids": np.asarray(tokens).astype(np.uint16)}
            # Process any remaining tokens.
            if sample_buffer:
                first_token_with_space = self.tokenizer.encode(" " + self.tokenizer.decode(sample_buffer[0]), add_special_tokens=False)
                tokens = self.bos_tokens + first_token_with_space + sample_buffer[1:] + self.eos_tokens
                n_tokens += len(tokens)
                yield {"input_ids": np.asarray(tokens).astype(np.uint16)}
        self.token_counts = n_tokens


def generate_samples(loader: DataLoader, 
                    #  truncate_num_samples: Optional[int] = None
                     ) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            yield {k: v[idx] for k, v in batch.items()}

def main(args: Namespace) -> None:
    if args.concat_tokens is None:
        raise ValueError("removed no concat functionality")
    
    if args.row_count:
        print(f"Processing csv: {args.dataset} (expected {args.row_count} rows)")
    else:
        print(f"Processing csv: {args.dataset}")
    csv_file = args.dataset
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, add_prefix_space=True)
    tokenizer.model_max_length = int(1e30)
    
    # Create a ConcatTokensDataset for just this CSV file.
    dataset = ConcatTokensDataset(
        csv_file=csv_file,
        tokenizer=tokenizer,
        max_length=args.concat_tokens,
        bos_text=args.bos_text,
        eos_text=args.eos_text,
        no_wrap=args.no_wrap,
    )

    def dict_collate_fn(batch):
        # Assuming each sample in the batch is a dictionary
        return {key: [d[key] for d in batch] for key in batch[0]}
    
    loader = DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=512,
        collate_fn=dict_collate_fn,
    )

    samples = generate_samples(loader)
    
    print(f"Writing to MDS for file {args.dataset}...")
    mds_output_dir = os.path.join(args.out_dir, os.path.basename(args.dataset).rsplit("_", 1)[0])
    with MDSWriter(columns={"input_ids": 'ndarray:uint16'}, out=mds_output_dir, compression=args.compression) as out:
        if args.row_count:
            progress = tqdm(samples, desc=f"Writing to {mds_output_dir}", total=args.row_count)
        else:
            progress = tqdm(samples, desc=f"Writing to {mds_output_dir}")

        for sample in progress:
            out.write(sample)

    with open(args.out_token_counts, "a") as file:
        file.write(f"{args.dataset} {dataset.token_counts:,}\n")

if __name__ == "__main__":
    main(parse_args())