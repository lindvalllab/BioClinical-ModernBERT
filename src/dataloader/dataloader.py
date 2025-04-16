from datasets import load_dataset

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
        return {"text": text, "label": label}