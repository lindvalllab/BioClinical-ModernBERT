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
        # batch["label"] is a list of lists, one per example.
        new_labels = []
        for label_list in batch["label"]:
            # If "none" (7) is present, set the labels to an empty multi-hot vector:
            if 7 in label_list:
                new_labels.append([0] * self.num_labels)
            else:
                binary_vector = [0] * self.num_labels
                for lab in label_list:
                    # For labs before 7, use them as is.
                    if lab < 7:
                        binary_vector[lab] = 1
                    # For labs after 7, subtract 1 because the "none" slot is removed.
                    elif lab > 7:
                        binary_vector[lab - 1] = 1
                new_labels.append(binary_vector)
        # Save the new multi-label encoding in a column named "labels"
        batch["labels"] = new_labels
        return batch