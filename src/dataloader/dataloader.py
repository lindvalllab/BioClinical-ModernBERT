import os
import pandas as pd
import torch
import random
from functools import partial
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent        # .../project/src/dataloader
PROJECT_ROOT = HERE.parent.parent             # .../project

def get_data(name):
    if name == "HOC":
        return HOC()
    elif name == "MedQA":
        return MedQA()
    elif name == "Phenotype":
        return Phenotype()
    elif name == "ChemProt":
        return ChemProt()
    elif name == "FactEHR":
        return FactEHR()
    else:
        raise NotImplementedError(f"Dataset {name} not implemented.")

class HOC:
    def __init__(self):
        self.class_names = [
            "evading growth suppressors",
            "tumor promoting inflammation",
            "enabling replicative immortality",
            "cellular energetics",
            "resisting cell death",
            "activating invasion and metastasis",
            "genomic instability and mutation",
            "none",
            "inducing angiogenesis",
            "sustaining proliferative signaling",
            "avoiding immune destruction",
        ]
        self.is_entailment = False
        self.num_labels = len(self.class_names) - 1 # drop "none", so 10 classes
        self.problem_type = "multi_label_classification"
        self.dataset = self.preprocess_data()

    def preprocess_data(self):
        ds = load_dataset("qanastek/HoC", trust_remote_code=True)
        ds_processed = ds.map(self.convert_labels, batched=True)
        columns_to_remove = ["document_id", "label"]
        ds_processed = ds_processed.remove_columns(columns_to_remove)
        return ds_processed

    def convert_labels(self, batch):
        new_labels = []
        for label_list in batch["label"]:
            if 7 in label_list:
                # Create a vector of zeros, and cast each element to float
                new_labels.append([0.0] * self.num_labels)
            else:
                binary_vector = [0.0] * self.num_labels  # start with float zeros
                for lab in label_list:
                    if lab < 7:
                        binary_vector[lab] = 1.0
                    elif lab > 7:
                        binary_vector[lab - 1] = 1.0
                new_labels.append(binary_vector)
        batch["labels"] = new_labels
        return batch
    

class Phenotype:
    def __init__(self):
        # The phenotype columns (exclude the "NONE" column as it only indicates "no phenotype")
        self.class_names = [
            'ADVANCED.CANCER', 'ADVANCED.HEART.DISEASE', 'ADVANCED.LUNG.DISEASE',
            'ALCOHOL.ABUSE', 'CHRONIC.NEUROLOGICAL.DYSTROPHIES', 'CHRONIC.PAIN.FIBROMYALGIA',
            'DEMENTIA', 'DEPRESSION', 'DEVELOPMENTAL.DELAY.RETARDATION', 'NON.ADHERENCE',
            'OBESITY', 'OTHER.SUBSTANCE.ABUSE',
            'SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS', 'UNSURE'
        ]
        self.is_entailment = False
        self.num_labels = len(self.class_names)
        self.problem_type = "multi_label_classification"
        self.cache_dir = os.path.join(f"{PROJECT_ROOT}/data/processed/phenotype")
        self.dataset = self.preprocess_data()

    def preprocess_data(self):
        # if we've cached already, just load
        if os.path.isdir(self.cache_dir):
            return DatasetDict.load_from_disk(self.cache_dir)
        # 1) Load the annotation and notes CSVs
        df_ann = pd.read_csv(f"{PROJECT_ROOT}/data/raw/phenotype/ACTdb102003.csv")
        df_mimic = pd.read_csv(f"{PROJECT_ROOT}/data/raw/phenotype/NOTEEVENTS.csv")

        # 2) Filter out inconsistent rows (NONE=1 & any phenotype=1)
        mask = ~((df_ann['NONE'] == 1) & (df_ann[self.class_names].sum(axis=1) > 0))
        df_ann = df_ann[mask].copy()

        # 3) Build the multi-hot "labels" vector of length len(class_names)
        df_ann['labels'] = df_ann.apply(self.make_multi_hot, axis=1)

        # 4) Merge with the text from NOTEEVENTS
        df = df_ann.merge(
            df_mimic[['ROW_ID', 'TEXT']],
            on='ROW_ID', how='inner'
        )[['TEXT', 'labels']]

        # 5) Rename for HF Dataset
        df = df.rename(columns={'TEXT': 'text'})

        # 6) Convert to HF Dataset and split
        ds = Dataset.from_pandas(df.reset_index(drop=True))

        # first split off 20% for test
        split1 = ds.train_test_split(test_size=0.2, seed=42)
        train_val = split1['train']
        test_ds   = split1['test']
        # then split train_val (10% of total)
        split2 = train_val.train_test_split(test_size=0.125, seed=42)
        train_ds = split2['train']
        val_ds   = split2['test']

        dataset_dict = DatasetDict({
            'train': train_ds,
            'validation': val_ds,
            'test': test_ds
        })

        # 6) cache to disk
        os.makedirs(self.cache_dir, exist_ok=True)
        dataset_dict.save_to_disk(self.cache_dir)

        return dataset_dict
    
    def make_multi_hot(self, row):
        # if NONE=1, then no phenotype => all zeros
        if row['NONE'] == 1:
            return [0.0] * self.num_labels
        else:
            # cast each entry to float
            return [float(x) for x in row[self.class_names].tolist()]


class ChemProt:
    def __init__(self):
        self.is_entailment = False
        self.problem_type = "single_label_classification"
        # where to cache the processed HuggingFace dataset
        self.cache_dir = os.path.join(f"{PROJECT_ROOT}/data/processed/ChemProt")
        # load (or preprocess & cache)
        self.dataset = self.preprocess_data()
        # after dataset is ready, set class names and num_labels
        self.class_names = self.dataset["train"].features["labels"].names
        self.num_labels = len(self.class_names)

    def preprocess_data(self):
        # 1) load from cache if available
        if os.path.isdir(self.cache_dir):
            return DatasetDict.load_from_disk(self.cache_dir)

        # 2) read train/dev/test TSVs
        splits = {}
        for split in ["train", "dev", "test"]:
            path = os.path.join(f"{PROJECT_ROOT}/data/raw/ChemProt/{split}.tsv")
            df = pd.read_csv(path, sep="\t")
            # rename columns
            df = df.rename(columns={"sentence": "text", "label": "label_str"})
            splits[split] = df

        # 4) convert each DataFrame to a Dataset, mapping labels
        ds_splits = {}
        for split, df in splits.items():
            ds = Dataset.from_pandas(df, preserve_index=False)
            # cast the string column to ClassLabel
            ds = ds.class_encode_column(
                column="label_str"
            )
            # now the label column is named "label_str" → rename to "labels"
            ds = ds.rename_column("label_str", "labels")
            ds_splits[split] = ds

        # 5) assemble DatasetDict, mapping "dev"→"validation"
        dataset_dict = DatasetDict({
            "train": ds_splits["train"],
            "validation": ds_splits["dev"],
            "test": ds_splits["test"]
        })

        # 6) cache to disk
        os.makedirs(self.cache_dir, exist_ok=True)
        dataset_dict.save_to_disk(self.cache_dir)

        return dataset_dict
    

class FactEHR:
    def __init__(self):
        self.cache_dir = os.path.join(f"{PROJECT_ROOT}/data/processed/factehr")
        self.is_entailment = True
        self.problem_type = "single_label_classification"
        self.dataset = self.preprocess_data()
        self.class_names = ["No", "Yes"]
        self.num_labels = len(self.class_names)

    def preprocess_data(self):
        # if we've cached already, just load
        if os.path.isdir(self.cache_dir):
            return DatasetDict.load_from_disk(self.cache_dir)
        
        # 1) Load the CSV file
        df = pd.read_csv(f"{PROJECT_ROOT}/data/raw/FactEHR/factehr_dev_set.csv")

        # 2) Rename columns for HF Dataset
        df = df.rename(columns={'premise': 'premise', 'hypothesis': 'hypothesis', 'human_label': 'labels'})

        # 3) Stratified split to maintain the distribution of labels
        train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])
        train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42, stratify=train_val_df['labels'])

        # Convert to HF Dataset
        train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
        test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

        dataset_dict = DatasetDict({
            'train': train_ds,
            'validation': val_ds,
            'test': test_ds
        })

        # 4) cache to disk
        os.makedirs(self.cache_dir, exist_ok=True)
        dataset_dict.save_to_disk(self.cache_dir)

        return dataset_dict
    

class MedQA:
    """Adapted from https://github.com/AnswerDotAI/ModernBERT-Instruct-mini-cookbook"""
    def __init__(self):
        # For MedQA, we have four options (A, B, C, D).
        self.problem_type = "mask_qa"
        self.dataset = self.preprocess_data()
        self.dummy_examples = True
        self.mlm_probability = 0.3
        self.true_mlm_proportions = 0.2

    def preprocess_data(self):
        # Load the MedQA dataset.
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", trust_remote_code=True)
        # Create a validation split: split the train split into train (80%) and validation (20%) using seed 42.
        split = ds["train"].train_test_split(test_size=0.2, seed=42)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]
        # Add the prompt
        ds["train"] = ds["train"].map(self.format_prompt, fn_kwargs={"is_test": False})
        ds["validation"] = ds["validation"].map(self.format_prompt, fn_kwargs={"is_test": False})
        ds["test"] = ds["test"].map(self.format_prompt, fn_kwargs={"is_test": True})
        # Remove columns that are no longer needed.
        remove_columns = ["question", "answer", "options", "meta_info", "answer_idx", "metamap_phrases"]
        ds = ds.remove_columns(remove_columns)
        return ds

    def format_prompt(self, example, is_test):
        mc_prefix = (
            "You will be given a medical question as well as a list of options. "
            "Read the question carefully and select the right answer from the list.\n"
            "QUESTION:\n"
        )
        text = mc_prefix + example["question"].strip() + "\nCHOICES:\n"
        text += "\n".join([f"- {k}: {v}" for k, v in sorted(example["options"].items())])
        if is_test:
            text += f"\nANSWER:\nAnswer: [unused0] [MASK]"
            return {"text": text, "label": example['answer_idx']}
        text += f"\nANSWER:\nAnswer: [unused0] {example['answer_idx']}" 
        return {"text": text}
    
    def tokenize_train_eval_datasets(self, tokenizer, max_length):
        fn_tokenize_train = partial(self.tokenize_split, tokenizer=tokenizer, max_length=max_length, is_eval=False)
        fn_tokenize_eval = partial(self.tokenize_split, tokenizer=tokenizer, max_length=max_length, is_eval=True)
        self.dataset["train"] = self.dataset["train"].map(fn_tokenize_train)
        self.dataset["validation"] = self.dataset["validation"].map(fn_tokenize_eval)
    
    def tokenize_split(self, example, tokenizer, max_length, is_eval):
        inputs = example["text"]

        tokenized_input = tokenizer(
            inputs,
            padding='max_length', 
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        if self.dummy_examples:
            input_ids = tokenized_input['input_ids'][0]
            attention_mask = tokenized_input['attention_mask'][0]
        else:
            # Clone to avoid modifying the original tensors.
            input_ids = tokenized_input['input_ids'][0].clone()
            attention_mask = tokenized_input['attention_mask'][0].clone()

        labels = torch.full_like(input_ids, -100)
        
        # Decide if we use full MLM masking or only mask the last non-special token.
        do_real_mlm = random.random() < self.true_mlm_proportions

        if do_real_mlm and not is_eval:
            # Create a probability matrix for MLM masking.
            probability_matrix = torch.full_like(input_ids, self.mlm_probability, dtype=torch.float)
            special_tokens_mask = torch.tensor(
                tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True),
                dtype=torch.bool
            )
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
            input_ids[masked_indices] = tokenizer.mask_token_id
            labels[masked_indices] = tokenized_input['input_ids'][0][masked_indices]
        else:
            # Only mask the last token before the final [SEP] (or last token with attention).
            sep_positions = (input_ids == tokenizer.sep_token_id).nonzero()
            if len(sep_positions) > 0:
                last_non_sep = sep_positions[-1].item() - 1
            else:
                last_non_sep = (attention_mask == 1).nonzero()[-1].item()
                
            original_token = input_ids[last_non_sep].clone()
            input_ids[last_non_sep] = tokenizer.mask_token_id
            labels[last_non_sep] = original_token
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }