# BioClinical ModernBERT

Repository for [BioClinical ModernBERT: A State-of-the-Art Long-Context Encoder for Biomedical and Clinical NLP](https://arxiv.org/abs/2506.10896) (preprint).

**This is the research repository for BioClinical ModernBERT. If you are looking to use our models, please refer to our [🤗 Collection](https://huggingface.co/collections/thomas-sounack/bioclinical-modernbert-681b824d12b9b6899841f8c7).**

This repository contains:
- The configuration files used to train the family of BioClinical ModernBERT models
- The performance benchmarking code
- The inference speed benchmarking code

While this repository does not include the full continued pretraining code (which was run using the [ModernBERT repo](https://github.com/AnswerDotAI/ModernBERT)), it does provide the configuration files needed to replicate the process. If you’re familiar with the ModernBERT codebase, you have everything required to get started right away using our [training checkpoints](https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-checkpoints). For those who prefer a walkthrough, we’ll be releasing a step-by-step guide soon.

## Configuration files

The folder `pretraining_configs` contains the configuration files used during the pretraining of BioClinical ModernBERT.

- The subfolder `phase1` contains the base and large configuration files for the general phase, where the models are trained on both the biomedical and the clinical data.
- The subfolder `phase2` contains the base and large configuration files for the specialization phase, where the models are trained on the clinical data only. We also provide the configuration files for Bio ModernBERT as `_phase2_bio_base` and `_phase2_bio_large`, which underperformed in our testing. Please refer to our paper for more details.

## Performance benchmarking

The script `main.py` can be used to fine-tune and evaluate encoders on a downstream tasks.

### Arguments:
* `dataset`: Name of the downstream task (Phenotype, ChemProt, DEID, COS or SocialHistory). You can implement more tasks in `dataloader.py` if needed. **Required**.
* `model`: HF Model to evaluate (e.g. answerdotai/ModernBERT-base). Can be a local path or a HF repo. **Required**.
* `--lr`: Learning rate for training. Optional, defaults to `2e-5`.
* `--wd`: Weight decay for training. Optional, defaults to `0.01`.
* `--epochs`: Number of training epochs. Optional, defaults to `3`.
* `--seed`: Random seed for reproducibility. Optional, defaults to `42`.
* `--batch_size`: Batch size per device for training and evaluation. Optional, defaults to `16`.
* `--accumulation_steps`: Gradient accumulation step. Optional, defaults to `1`.