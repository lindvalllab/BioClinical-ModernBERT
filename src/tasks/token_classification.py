from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from functools import partial
from src.metrics.metrics import compute_metrics_token_classification

class TokenClassificationTrainer:
    def __init__(self, device, model_checkpoint, data_wrapper, training_args, checkpoint_dir):
        self.device = device
        self.model_checkpoint = model_checkpoint
        self.ds = data_wrapper.dataset
        self.training_args = training_args
        self.checkpoint_dir = checkpoint_dir
        self.id2label = data_wrapper.id2label
        self.label2id = data_wrapper.label2id

        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=data_wrapper.num_labels
        )
        self.model.to(self.device)
        
        # Preprocess the dataset: tokenize and align labels.
        self.ds = self.ds.map(self.tokenize_and_align_labels, batched=True)
        # Set up dynamic padding via a data collator.
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.trainer = None
        self.test_results = None

    def tokenize_and_align_labels(self, examples):
        """
        Tokenizes examples and realigns the token-level labels.
        Special tokens (e.g. [CLS] and [SEP]) get a label of -100 so they are ignored in the loss.
        
        Args:
            examples (dict): A batch from the dataset containing "tokens" and "ner_tags".
        
        Returns:
            dict: The tokenized inputs with a new "labels" key.
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
        )
        
        all_labels = []
        # Loop over each example.
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # Label for the first token of the word.
                    label_ids.append(labels[word_idx])
                else:
                    # Subsequent tokens in a split word get label -100.
                    label_ids.append(-100)
                previous_word_idx = word_idx
            all_labels.append(label_ids)
        
        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    def train(self):
        """
        Sets up TrainingArguments and creates a Trainer for token classification.
        Then starts the training process.
        """
        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
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
            compute_metrics=partial(compute_metrics_token_classification, id2label=self.id2label)
        )
        self.trainer.train()

    def evaluate(self):
        """
        Evaluates the model on the test split and stores the test results.
        """
        self.test_results = self.trainer.evaluate(self.ds["test"])