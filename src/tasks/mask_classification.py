from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from src.metrics.metrics import evaluate_mmlu, compute_metrics_mlm_for_training

class MaskClassificationTrainer():
    def __init__(self, device, model_checkpoint, data_wrapper, training_args, checkpoint_dir):
        self.device = device
        self.model_checkpoint = model_checkpoint
        self.training_args = training_args
        self.checkpoint_dir = checkpoint_dir

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, add_prefix_space=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_checkpoint
        )
        self.model.to(self.device)
        # fix for emilyalsentzer/Bio_ClinicalBERT
        self.max_length = self.tokenizer.model_max_length if self.tokenizer.model_max_length < 10000 else 512
        
        data_wrapper.tokenize_train_eval_datasets(self.tokenizer, self.max_length)
        self.ds = data_wrapper.dataset

        self.trainer = None
        self.test_results = None

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            save_total_limit=1,
            report_to="none",
            # load_best_model_at_end=True,
            # metric_for_best_model="accuracy",
            # greater_is_better=True,
            # eval_accumulation_steps=1,
            **self.training_args
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.ds["train"],
            eval_dataset=self.ds["validation"],
            tokenizer=self.tokenizer,
            # compute_metrics=compute_metrics_mlm_for_training
        )
        self.trainer.train()
    
    def evaluate(self):
        model = self.trainer.model
        model.eval()
        self.test_results = evaluate_mmlu(model, self.tokenizer, self.ds["test"])