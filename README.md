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
  
## Datasets

- Twitter-Tennis is loaded through PyTorch Geometric Temporal.
- Reddit-Body and Wiki-RfA are available in the shared Google Drive folder:
  https://drive.google.com/drive/folders/1_ZEbWoVZVUi8O5bWCR49U6AX2MrGx56E?usp=sharing
Before running the Reddit-Body and Wiki-RfA scripts, set DATA_PATH in each script to the path of its downloaded dataset file.
## Usage

Before running the Reddit and Wiki-RfA scripts, update DATA_PATH according to the dataset location on your computer.

```bash
python STG_GAN_twitter.py
python reddit_stg_gan.py
python wiki_rfa_stg_gan.py
