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
        # List of phenotype columns (excluding 'NONE')
        self.phenotype_cols = [
            'ADVANCED.CANCER', 'ADVANCED.HEART.DISEASE', 'ADVANCED.LUNG.DISEASE',
            'ALCOHOL.ABUSE', 'CHRONIC.NEUROLOGICAL.DYSTROPHIES', 'CHRONIC.PAIN.FIBROMYALGIA',
            'DEMENTIA', 'DEPRESSION', 'DEVELOPMENTAL.DELAY.RETARDATION', 'NON.ADHERENCE',
            'OBESITY', 'OTHER.SUBSTANCE.ABUSE',
            'SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS', 'UNSURE'
        ]
        # The full list of columns including "NONE"
        self.all_phenotype_cols = self.phenotype_cols + ["NONE"]
        self.num_labels = len(self.all_phenotype_cols)
        self.problem_type = "multi_label_classification"
        self.dataset = self.preprocess_data()

    def preprocess_data(self):
        # Load the annotation CSV and the corresponding notes CSV from local files.
        df_ann = pd.read_csv("../../data/phenotype/ACTdb102003.csv")
        df_mimic = pd.read_csv("../../data/phenotype/NOTEEVENTS.csv")
        
        # Filter out rows where 'NONE' is 1 but there are other phenotype flags active.
        valid_rows_mask = ~((df_ann['NONE'] == 1) & (df_ann[self.phenotype_cols].sum(axis=1) > 0))
        df_ann_clean = df_ann[valid_rows_mask].copy()
        
        # Create a multi-hot label vector.
        df_ann_clean['label'] = df_ann_clean.apply(self.create_label, axis=1)
        
        # Merge with the mimic notes using the ROW_ID key.
        df_merged = df_ann_clean.merge(df_mimic[['ROW_ID', 'TEXT']], on='ROW_ID', how='inner')
        
        # Keep only the TEXT and label columns.
        df_final = df_merged[['TEXT', 'label']]
        
        # Create a human-readable string for the label.
        def multi_hot_to_str(label_vector, names):
            # Only include names from phenotype_cols (not 'NONE') when a flag is active.
            active = [name for val, name in zip(label_vector, self.phenotype_cols) if val == 1]
            return ",".join(active)
        df_final['label_str'] = df_final['label'].apply(lambda vec: multi_hot_to_str(vec, self.phenotype_cols))
        
        # Rename the TEXT column to "text" (and optionally rename 'label' to 'labels').
        df_final = df_final.rename(columns={'TEXT': 'text', 'label': 'labels'})
        
        # Create a unique identifier for each example.
        df_final = df_final.reset_index().rename(columns={'index': 'guid'})
        
        # Convert the pandas DataFrame to a Hugging Face Dataset.
        ds = Dataset.from_pandas(df_final)
        
        # Split the dataset into train (80%), validation (10%), and test (10%).
        # First, reserve 20% for the test set.
        ds_split = ds.train_test_split(test_size=0.2, seed=42)
        ds_test = ds_split["test"]
        ds_train_val = ds_split["train"]
        # From the remaining 80%, reserve 12.5% for validation (which makes overall 10% validation).
        ds_train_val = ds_train_val.train_test_split(test_size=0.125, seed=42)
        ds_train = ds_train_val["train"]
        ds_validation = ds_train_val["test"]
        
        dataset_dict = DatasetDict({
            "train": ds_train,
            "validation": ds_validation,
            "test": ds_test
        })  
        
        return dataset_dict

    def create_label(self, row):
        if row['NONE'] == 1:
            return [0] * self.num_labels
        else:
            return row[self.all_phenotype_cols].tolist()