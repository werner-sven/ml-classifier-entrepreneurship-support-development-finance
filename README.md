# Measuring Entrepreneurship-Support Development Finance (ESDF)

Identifying entrepreneurship-support development finance (ESDF) from textual project descriptions in OECD CRS development finance project-level data using a fine-tuned transformer classifier.

This repository contains the underlying code for the machine learning classifier of the scientific paper "From Portfolio Targeting to Entrepreneurial Dynamics: Evidence from a New Measure of Entrepreneurship-Support Development Finance" by Sven Werner and Philipp Trotter.

How to work with this repository:

# Installation

Python Interpreter: Python 3.12+

Install required packages:

```shell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

# Data

* Download preprocessed OECD CRS data: https://zenodo.org/records/18498599/files/crs_en.parquet
* Save Parquet File under `data/01_raw/`
* The raw development finance data is publicly available on the OECD's data explorer: https://data-explorer.oecd.org/
* Data paths are configured in `config.json`.

# Required inputs

1. **CRS input PARQUET** (`FILES.CRS_PARQUET`, default: `data/01_raw/crs_en.parquet`)

   * Must contain a text column `raw_text` which represents the English input strin**g** used for classification (e.g., concatenated title + descriptions, with non-English text translated upstream). 
   * The preprocessed dataset needs to be stored under `data/01_raw/`. 

2. **Gold labels** (`FILES.GOLD_LABELED`, default: `data/01_raw/gold_labeled.xlsx`)

   * Our handcoded gold data is contained in the repo under the respective link. 
   * File must contain columns:

     * `ProjectDesc` (text)
     * `gold_es` (binary label 0/1)


# Usage
* You can either run the full pipeline via python code/run_all.py or execute each step (01..06) separately (see below).

* Note: The full pipeline is computationally heavy (processing a ~5.0 mio row dataset, fine-tuning three transformer models, and predicting ~1.5 mio unique project descriptions). Ensure sufficient computing power is available.

* To replicate only model fine-tuning and testing, run steps 01–04 only.

* For repeated runs, ensure that the folder output/02_best_model is empty to avoid confusion between model selection between multiple runs.


## Run full pipeline

Run all steps sequentially:

```shell
python code/run_all.py
```


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
