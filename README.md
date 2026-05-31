# JointCL Political Stance Detection

This project implements a JointCL-based political stance detection pipeline for classifying LLM-generated political summaries as left, right, or neutral.

## Method

The pipeline uses:

- BERT-based text encoding
- supervised contrastive stance learning
- FAISS k-means prototype clustering
- prototype contrastive loss
- final stance classification

## Pipeline

1. Load custom train/dev/test CSV files
2. Tokenize summary-target pairs with BERT
3. Encode examples using BERT
4. Construct stance prototypes using FAISS clustering
5. Train with classification loss, stance contrastive loss, and prototype contrastive loss
6. Select best checkpoints using development accuracy and macro F1
7. Evaluate on the test set and save reports

## Main File

`run_vast.py`

## How to Run

```bash
pip install -r requirements.txt
python run_vast.py \
  --train_dir ./custom_jointcl_data/train_sample.csv \
  --dev_dir ./custom_jointcl_data/dev_sample.csv \
  --test_dir ./custom_jointcl_data/test_sample.csv \
  --output_par_dir ./outputs
