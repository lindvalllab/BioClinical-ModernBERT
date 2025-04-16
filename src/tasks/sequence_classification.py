from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from src.metrics.metrics import compute_metrics_multi_label_classification, compute_metrics_single_label_classification

class SequenceClassificationTrainer():
    def __init__(self, device, model_checkpoint, data_wrapper, training_args, checkpoint_dir):
        self.device = device
        self.model_checkpoint = model_checkpoint
        self.ds = data_wrapper.dataset
        self.problem_type = data_wrapper.problem_type
        self.training_args = training_args
        self.checkpoint_dir = checkpoint_dir

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, add_prefix_space=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=data_wrapper.num_labels,
            problem_type=self.problem_type
        )
        self.model.to(self.device)
        # fix for emilyalsentzer/Bio_ClinicalBERT
        self.max_length = self.tokenizer.model_max_length if self.tokenizer.model_max_length < 10000 else 512
        self.ds = self.ds.map(self.tokenize, batched=True, remove_columns=["text"])
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.trainer = None
        self.test_results = None
    
    def tokenize(self, batch):
        return self.tokenizer(batch["text"], truncation=True, max_length=self.max_length)
    
    def get_compute_metrics(self):
        if self.problem_type == "multi_label_classification":
            return compute_metrics_multi_label_classification
        elif self.problem_type == "single_label_classification":
            return compute_metrics_single_label_classification
        else:
            raise NotImplementedError(f"No compute metrics function for {self.problem_type} has been implemented")

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="weighted_f1",
            save_total_limit=1,
            report_to="none",
            **self.training_args
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.ds["train"],
            eval_dataset=self.ds["validation"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.get_compute_metrics(),
        )
        self.trainer.train()
    
    def evaluate(self):
        self.test_results = self.trainer.evaluate(self.ds["test"])