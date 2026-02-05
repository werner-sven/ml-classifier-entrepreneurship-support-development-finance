# -----------------------------------------------------------------------------
# run_all.py
# Sequentially execute all pipeline scripts (00..06) using CODE_DIR from config;
# stops immediately on first failure (check=True).
# Reads:  config.json (DIRS.CODE_DIR) + each script’s inputs
# Writes: all outputs produced by 00..06
# -----------------------------------------------------------------------------


#%% -----------------------------------------------------------------------------#
# Config & Paths
# -----------------------------------------------------------------------------#

import sys
import subprocess

from utils import load_config, resolve_path

PROJECT_ROOT, cfg = load_config(config_name="config.json")

# All script paths are resolved relative to PROJECT_ROOT via config
CODE_DIR = resolve_path(PROJECT_ROOT, cfg["DIRS"]["CODE_DIR"])

SCRIPTS = [
    "00_ingest_crs.py",
    "01_extract_unique_projectdesc.py",
    "02_split_gold_data.py",
    "03_train_models_eval.py",
    "04_test_best_model.py",
    "05_predict_unique_projectdesc.py",
    "06_merge_predictions_to_crs.py",
]

script_paths = [CODE_DIR / s for s in SCRIPTS]


#%% -----------------------------------------------------------------------------#
# Run
# -----------------------------------------------------------------------------#

print(f"Run all | code_dir={CODE_DIR}")

for p in script_paths:
    subprocess.run([sys.executable, str(p)], check=True)
