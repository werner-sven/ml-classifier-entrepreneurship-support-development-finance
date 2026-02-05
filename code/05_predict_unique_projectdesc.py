# -----------------------------------------------------------------------------
# 05_predict_unique_projectdesc.py
# Predict best model on unique ProjectDesc parquet (optionally sampled); supports
# resume by skipping text_key already present in existing predictions parquet.
# Outputs ES_prob (float) + ES (0/1) per text_key.
# Reads:  FILES.UNIQUE_PROJECTDESC, FILES.BEST_MODEL_JSON (+ model dir referenced therein)
# Writes: FILES.PREDICTIONS_UNIQUE (append/overwrite combined on resume)
# -----------------------------------------------------------------------------


#%% -----------------------------------------------------------------------------#
# Config & Paths
# -----------------------------------------------------------------------------#

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import (
    load_config,
    norm_text,
    make_text_key,
    read_parquet_pandas,
    resolve_path,
    safe_write_parquet_pandas,
)

PROJECT_ROOT, cfg = load_config(config_name="config.json")

SEED = int(cfg.get("SEED", 42))
torch.manual_seed(SEED)

PRED_BATCH_SIZE = int(cfg.get("PREDICT", {}).get("BATCH_SIZE", 64))
RESUME = bool(cfg.get("PREDICT", {}).get("RESUME", True))

# Keep sample logic (for testing)
FIRST_N_ROWS = cfg.get("FIRST_N_ROWS", None)          # set to int for quick tests
SAMPLE_LIMIT = cfg.get("SAMPLE_LIMIT", None)          # set to int for random sample

PARQUET_COMPRESSION = cfg.get("PARQUET_COMPRESSION", "snappy")

UNIQUE_IN = resolve_path(PROJECT_ROOT, cfg["FILES"]["UNIQUE_PROJECTDESC"])
path_best_json = resolve_path(PROJECT_ROOT, cfg["FILES"]["BEST_MODEL_JSON"])
path_out_preds = resolve_path(PROJECT_ROOT, cfg["FILES"]["PREDICTIONS_UNIQUE"])

COL_TEXT = "ProjectDesc"
COL_KEY = "text_key"


#%% -----------------------------------------------------------------------------#
# Helpers (lean)
# -----------------------------------------------------------------------------#

def _softmax_pos_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]


#%% -----------------------------------------------------------------------------#
# Run Script
# -----------------------------------------------------------------------------#

df = read_parquet_pandas(UNIQUE_IN)

assert COL_TEXT in df.columns

# Enforce canonical normalization (idempotent) + fail on empty
df[COL_TEXT] = df[COL_TEXT].astype("string").fillna("").map(norm_text)
assert (df[COL_TEXT] != "").all()

# Ensure stable join key exists
if COL_KEY not in df.columns:
    df[COL_KEY] = df[COL_TEXT].map(make_text_key)

# Sample logic (keep in place for testing)
if FIRST_N_ROWS is not None:
    df = df.head(int(FIRST_N_ROWS)).copy()
elif SAMPLE_LIMIT is not None:
    sample_n = int(SAMPLE_LIMIT)
    head_n = sample_n * 10000
    df_head = df.head(head_n)
    df = df_head.sample(n=sample_n, random_state=SEED).copy()

# Load best model + threshold
with path_best_json.open("r", encoding="utf-8") as f:
    best = json.load(f)

thr = float(best.get("threshold_eval") or best.get("best_thr") or best.get("best_thr_eval"))

best_model_dir = Path(best["paths"]["best_model_dir"])
best_model_dir = best_model_dir if best_model_dir.is_absolute() else resolve_path(PROJECT_ROOT, best_model_dir)

max_length = int(best.get("hparams", {}).get("max_length") or cfg.get("TRAIN", {}).get("MAX_LENGTH", 256))

tokenizer = AutoTokenizer.from_pretrained(best_model_dir, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Resume: skip already predicted keys (if output exists)
df_prev = None
if RESUME and path_out_preds.exists():
    df_prev = read_parquet_pandas(path_out_preds)
    assert COL_KEY in df_prev.columns
    done_keys = set(df_prev[COL_KEY].astype("string").tolist())
    df_todo = df.loc[~df[COL_KEY].astype("string").isin(done_keys)].copy()
else:
    df_todo = df.copy()

texts = df_todo[COL_TEXT].astype(str).tolist()
probs = np.empty(len(texts), dtype=float)

for i in range(0, len(texts), PRED_BATCH_SIZE):
    batch_texts = texts[i : i + PRED_BATCH_SIZE]
    enc = tokenizer(
        batch_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        p = _softmax_pos_probs(out.logits).detach().cpu().numpy()

    probs[i : i + len(batch_texts)] = p

df_todo["prob_es"] = probs.astype(float)
df_todo["pred_es"] = (df_todo["prob_es"].to_numpy() >= thr).astype(int)

# Combine with previous predictions (if resume)
if df_prev is not None:
    df_out = pd.concat([df_prev, df_todo], ignore_index=True)
else:
    df_out = df_todo

safe_write_parquet_pandas(df_out, path_out_preds, compression=PARQUET_COMPRESSION, index=False)

print(f"Wrote: {path_out_preds} | n={len(df_out):,} | thr={thr:.3f}")
