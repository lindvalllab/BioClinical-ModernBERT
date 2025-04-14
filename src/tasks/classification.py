from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from src.metrics.metrics import compute_metrics_multi_label_classification

class ClassificationTrainer():
    def __init__(self, device, model, data_wrapper, training_args, checkpoint_dir):
        self.device = device
        self.model_checkpoint = model
        self.ds = data_wrapper.dataset
        self.problem_type = data_wrapper.problem_type
        self.training_args = training_args
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=data_wrapper.num_labels,
            problem_type=self.problem_type
        )
        self.model.to(self.device)
        self.max_length = self.tokenizer.model_max_length
        self.ds = self.ds.map(self.tokenize, batched=True)
        self.compute_metrics = None
        self.trainer = None
        self.test_results = None
    
    def tokenize(self, batch):
        return self.tokenizer(batch["text"], padding="max_length", truncation=True, max_length=self.max_length)
    
    def get_compute_metrics(self):
        if self.problem_type == "multi_label_classification":
            return compute_metrics_multi_label_classification
        else:
            raise NotImplementedError(f"No compute metrics function for {self.problem_type} has been implemented")

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="weighted_f1",
            save_total_limit=1,
            **self.training_args
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.ds["train"],
            eval_dataset=self.ds["validation"],
            compute_metrics=self.get_compute_metrics(),
        )
        self.trainer.train()
    
    def evaluate(self):
        self.test_results = self.trainer.evaluate(self.ds["test"])