# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#comet-ml-project">CometML Project</a> •
  <a href="#demo-notebook">Demo Notebook</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a solution for homework 1 of the HSE DLA course

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

In this repo, Python 3.10 was used.

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model and reproduce the homework submission, run the following command:

```bash
python -u train.py \
    trainer.n_epochs=10 \
    trainer.epoch_len=6000 \
    dataloader.batch_size=64 \
    dataloader.num_workers=12 \
    model.hidden_size=256 \
    model.num_layers=12 \
    lr_scheduler.max_lr=0.0005 \
    text_encoder.tokenizer_path=data/tokenizers/bpe_128.json \
    text_encoder.tokenizer_text_dir=data/datasets/librispeech/train-clean-100 \
    text_encoder.vocab_size=128 \
    text_encoder.lm_model.model_name=EleutherAI/pythia-410m \
    text_encoder.lm_model.batch_size=32 \
    text_encoder.beam_search_njobs=12 \
    transforms.instance_transforms.train.audio.transforms.1.min_snr_in_db=10.0 \
    transforms.instance_transforms.train.audio.transforms.1.max_snr_in_db=25.0 \
    transforms.instance_transforms.train.audio.transforms.1.p=0.7 \
    transforms.instance_transforms.train.audio.transforms.2.min_snr_db=10.0 \
    transforms.instance_transforms.train.audio.transforms.2.max_snr_db=25.0 \
    transforms.instance_transforms.train.audio.transforms.2.p=0.5 \
    transforms.instance_transforms.train.audio.transforms.3.p=0.0
```

To run inference on the final model (evaluate the model or save predictions):

```bash
python inference.py \
    --config-name=inference_final \
    resources_dir=/path/to/asr_model \
    dataset_root=/path/to/dataset
```
where `asr_model` contains model checkpoint and tokenizer for BPE. They can be found [here](https://drive.google.com/file/d/1o4tfb3WCsIrpPvNtz4IlOKk3E5Jzbnjj/view?usp=drive_link), or downloaded with a script

```bash
pip install gdown
gdown 1o4tfb3WCsIrpPvNtz4IlOKk3E5Jzbnjj -O asr_model.zip
```

To calculate metrics:

```bash
python -m scripts.calc_metrics \
    --target-dir /path/to/dataset/transcriptions \
    --predicted-dir data/saved/predictions/test
```

## CometML Project

CometML project with the report can be found [here](https://www.comet.com/abramov-g-evg/hse-dla-asr/)

## Demo Notebook

Google Colab demo notebook can be found [here](https://colab.research.google.com/drive/12e1arghhqAX3tO6Hqf7IiMGHJe3yYkjQ?usp=sharing), or in a repo. It does not use the `inference_final` config and creates it on the spot.

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
