# STG-GAN

This repository provides the implementation of STG-GAN associated with the manuscript:

*“STG-GAN: An Integrated Spatio-Temporal Adversarial Framework for Anomaly Detection in Dynamic Social Networks.”*

## Available implementations

- STG_GAN_twitter.py: Twitter-Tennis dataset
- reddit_stg_gan.py: Reddit Hyperlinks dataset
- wiki_rfa_stg_gan.py: Wiki-RfA dataset

## Requirements

- Python 3
- PyTorch
- PyTorch Geometric
- PyTorch Geometric Temporal
- NumPy
- pandas
- scikit-learn

## Usage

The datasets are not included in this repository and must be downloaded separately from their original sources.

Before running the Reddit and Wiki-RfA scripts, update DATA_PATH according to the dataset location on your computer.

```bash
python STG_GAN_twitter.py
python reddit_stg_gan.py
python wiki_rfa_stg_gan.py
