# Measuring Entrepreneurship-Support Development Finance (ESDF)

Estimating entrepreneurship-support development finance (ESDF) from textual project descriptions in OECD CRS data using a fine-tuned transformer classifier.

This repository contains the classification code underlying the paper draft “(title TBD)” by Sven Werner and Philipp Trotter.

# Guideline

How to work with this repository.

## Installation

Python Interpreter: Python 3.10+

Install required packages:

```shell
pip install -r requirements.txt
```

## Data

Data paths are configured in `config.json`.

Download needed data (CRS input and hand-coded gold labels) from https://1drv.ms/f/c/eb405fe4e29e9bf9/IgCGBUgKhpyaRZI0Pcfy3m9YAQ-jSJtJn6bmuQNAo6LqAAA?e=ffaPD3 . Save those in folder "data/01_raw".

### Required inputs

1. **CRS input parquet** (`FILES.CRS_PARQUET`, default: `data/01_raw/crs_en.parquet`)

   * Must contain a text column `raw_text` which represents the English input strin**g** used for classification (e.g., concatenated title + descriptions, with non-English text translated upstream). Our preprocessed dataset is stored under the default input in this repo. Original data is publicly available on the OECD's data explorer in the download tab: https://shorturl.at/gnZJc 

2. **Gold labels** (`FILES.GOLD_LABELED`, default: `data/01_raw/gold_labeled.xlsx`)

   * The default input contains the file with our labelled data. File must contain columns:

     * `ProjectDesc` (text)
     * `gold_es` (binary label 0/1)

## Usage

## Run full pipeline

Run all steps sequentially:

```shell
python code/run_all.py
```

Note: Step 5 (using the finetuned classifier to predict the full dataset) is computationally heavy as there are >1.5 mio unique descriptions to predict. To only replicate model training and testing run steps 1-4. 
Further, for repeated runs, ensure that the folder output/02_best_model is empty to avoid confusion between model selection between multiple runs.


## Pipeline steps (01..06)

### 01 — Extract unique project descriptions

Builds a unique table of project descriptions to classify once per unique text:

```shell
python code/01_extract_unique_projectdesc.py
# writes: FILES.UNIQUE_PROJECTDESC
```

### 02 — Split gold-labeled data

Creates stratified train/eval/test splits from the labeled Excel file:

```shell
python code/02_split_gold_data.py
# writes: FILES.TRAIN_SPLIT, FILES.EVAL_SPLIT, FILES.TEST_SPLIT
```

### 03 — Train model candidates + select best

Trains a small hyperparameter grid, tunes a probability threshold on the eval split, and writes a `best_model.json` plus report tables:

```shell
python code/03_train_models_eval.py
# writes: output/01_runs/*, output/02_best_model/best_model.json, report tables to output/03_reports/
```

### 04 — Test best model

Evaluates the selected best model once on the held-out test split and exports tables/figures:

```shell
python code/04_test_best_model.py
# writes: test metrics + confusion matrix + examples and paper-style exports
```

### 05 — Predict on unique project descriptions

Runs the best model on the unique ProjectDesc table (supports resume) and writes predictions per `text_key`:

```shell
python code/05_predict_unique_projectdesc.py
# writes: FILES.PREDICTIONS_UNIQUE
```

### 06 — Merge predictions back into CRS

Merges unique-text predictions into the full CRS dataset using `text_key`:

```shell
python code/06_merge_predictions_to_crs.py
# writes: FILES.CRS_WITH_PREDICTIONS
```
