# BioClinical ModernBERT

Repository for [BioClinical ModernBERT: A State-of-the-Art Long-Context Encoder for Biomedical and Clinical NLP](https://arxiv.org/abs/2506.10896) (preprint).

**This is the research repository for BioClinical ModernBERT. If you are looking to use our models, please refer to our [🤗 Collection](https://huggingface.co/collections/thomas-sounack/bioclinical-modernbert-681b824d12b9b6899841f8c7).**


## Table of Contents
1. [Setup](#setup)
2. [Pretraining configuration files](#pretraining-configuration-files)
3. [Performance benchmarking code](#performance)
4. [Inference speed benchmarking code](#inference-speed)
5. [Reference](#reference)

While this repository does not include the full continued pretraining code (which was run using the [ModernBERT repo](https://github.com/AnswerDotAI/ModernBERT)), it does provide the configuration files needed to replicate the process. If you’re familiar with the ModernBERT codebase, you have everything required to get started right away using our [training checkpoints](https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-checkpoints). For those who prefer a walkthrough, we’ll be releasing a step-by-step guide soon.


## Setup

**⚠️ To reproduce our results, please ensure you run the scripts on a GPU compatible with Flash Attention 2.**

TODO

## Pretraining configuration files

The folder `pretraining_configs` contains the configuration files used during the pretraining of BioClinical ModernBERT.

- The subfolder `phase1` contains the base and large configuration files for the general phase, where the models are trained on both the biomedical and the clinical data.
- The subfolder `phase2` contains the base and large configuration files for the specialization phase, where the models are trained on the clinical data only. We also provide the configuration files for Bio ModernBERT as `_phase2_bio_base` and `_phase2_bio_large`, which underperformed in our testing. Please refer to our paper for more details.


## Performance

The script `main.py` can be used to fine-tune and evaluate encoders on a downstream tasks.

### Datasets
The datasets used in this repo need to be downloaded manually and added to the `data/raw` folder:
- Phenotype: [Physionet link](https://www.physionet.org/content/phenotype-annotations-mimic/1.20.03/). Note that this dataset also requires [MIMIC III](https://physionet.org/content/mimiciii/1.4/)'s NOTEEVENTS csv. 
- ChemProt: [BLUE Benchmark github release](https://github.com/ncbi-nlp/BLUE_Benchmark/releases/download/0.1/bert_data.zip)
- DEID: [Physionet link](https://www.physionet.org/content/deidentifiedmedicaltext/1.0/)
- COS: [Washington BioNLP link](https://depts.washington.edu/bionlp/data/corpora/files/events-COS-corpus.zip)
- SocialHistory: [Washington BioNLP link](https://depts.washington.edu/bionlp/data/corpora/files/SocialHistoryMTSamples.zip)

### Arguments:
* `--dataset`: Name of the downstream task (Phenotype, ChemProt, DEID, COS or SocialHistory). You can implement more tasks in `dataloader.py` if needed. **Required**.
* `--model`: HF Model to evaluate (e.g. thomas-sounack/BioClinical-ModernBERT-base). Can be a local path or a HF repo. **Required**.
* `--lr`: Learning rate for training. Optional, defaults to `2e-5`.
* `--wd`: Weight decay for training. Optional, defaults to `0.01`.
* `--epochs`: Number of training epochs. Optional, defaults to `3`.
* `--seed`: Random seed for reproducibility. Optional, defaults to `42`.
* `--batch_size`: Batch size per device for training and evaluation. Optional, defaults to `16`.
* `--accumulation_steps`: Gradient accumulation step. Optional, defaults to `1`.

For example:
```
python main.py --dataset Phenotype --model thomas-sounack/BioClinical-ModernBERT-base --lr 2e-5 --wd 0.01 --epochs 3 --seed 42 --batch_size 16 --accumulation_steps 1
```

For your convenience, we also provide the bash script `scripts/run_parallel.sh`. It can be called with the same hyperparameters as main.py. If a seed is provided, it is equivalent to main.py. Otherwise, it launches multiple training runs in parallel with different seeds, according to the list `seeds` in that script.

### Results exploration

The notebook `notebooks/downstream_results_exploration.ipynb` can be used to compare the fine-tuned models.


##  Inference speed

The script `multiprocess_bench.py` is used to measure the inference speed of each model. It is a modified version of ModernBERT's inference speed script ([see original](https://github.com/AnswerDotAI/ModernBERT/blob/8c57a0f01c12c4953ead53d398a36f81a4ba9e38/efficiency/multiprocess_bench.py)), where we add a third dataset size (medium) to compare our model with encoders that have a 4096 token input length.

It can be used with the following command:

```
python multiprocess_bench.py --model thomas-sounack/BioClinical-ModernBERT-base > BioClinical-ModernBERT-base_inference_times.log 2>&1
```


## Reference

If you use BioClinical ModernBERT in your work, whether it be this code, our models or our training checkpoints, please cite our [preprint](https://arxiv.org/abs/2506.10896):

```
@misc{sounack2025bioclinicalmodernbertstateoftheartlongcontext,
      title={BioClinical ModernBERT: A State-of-the-Art Long-Context Encoder for Biomedical and Clinical NLP}, 
      author={Thomas Sounack and Joshua Davis and Brigitte Durieux and Antoine Chaffin and Tom J. Pollard and Eric Lehman and Alistair E. W. Johnson and Matthew McDermott and Tristan Naumann and Charlotta Lindvall},
      year={2025},
      eprint={2506.10896},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.10896}, 
}
```