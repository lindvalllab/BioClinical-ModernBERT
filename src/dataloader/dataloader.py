from datasets import load_dataset

def get_data(name):
    if name == "HOC":
        return HOC()
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