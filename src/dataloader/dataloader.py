import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

def get_data(name):
    if name == "HOC":
        return HOC()
    elif name == "MedQA":
        return MedQA()
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
    

class MedQA:
    def __init__(self):
        # For MedQA, we have four options (A, B, C, D).
        self.class_names = ["A", "B", "C", "D"]
        self.num_labels = len(self.class_names)   # 4 classes for single-label classification
        self.problem_type = "single_label_classification"
        self.dataset = self.preprocess_data()

    def preprocess_data(self):
        # Load the MedQA dataset.
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", trust_remote_code=True)
        # Create a validation split: split the train split into train (80%) and validation (20%) using seed 42.
        split = ds["train"].train_test_split(test_size=0.2, seed=42)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]
        # Process each example to produce the 'text' and 'label' fields.
        ds = ds.map(self.process_example)
        # Remove columns that are no longer needed.
        remove_columns = ["question", "answer", "options", "meta_info", "answer_idx", "metamap_phrases"]
        ds = ds.remove_columns(remove_columns)
        return ds

    def process_example(self, example):
        # Concatenate the question with the options.
        question = example["question"]
        # 'options' is a dict; sort by key to preserve order A, B, C, D.
        formatted_options = ", ".join([f"{k}: {v}" for k, v in sorted(example["options"].items())])
        text = f"{question} The options are: {formatted_options}"
        # Map the answer (a letter) to a label (0=A, 1=B, 2=C, 3=D)
        mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        label = mapping[example["answer_idx"]]
        return {"text": text, "labels": label}
    

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
        self.num_labels = len(self.class_names)
        self.problem_type = "multi_label_classification"
        self.cache_dir = os.path.join("../../data/processed/phenotype")
        self.dataset = self.preprocess_data()

    def preprocess_data(self):
        # if we've cached already, just load
        if os.path.isdir(self.cache_dir):
            return DatasetDict.load_from_disk(self.cache_dir)
        # 1) Load the annotation and notes CSVs
        df_ann = pd.read_csv("../../data/raw/phenotype/ACTdb102003.csv")
        df_mimic = pd.read_csv("../../data/raw/phenotype/NOTEEVENTS.csv")

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
                return [0] * self.num_labels
            else:
                return row[self.class_names].astype(int).tolist()