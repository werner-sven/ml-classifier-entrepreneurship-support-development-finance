# -----------------------------------------------------------------------------
# 06_merge_predictions_to_crs.py
# Merge predictions back into CRS on text_key; sanity-check equivalence vs merging
# on ProjectDesc (slice); fill unmatched rows as NO_ES (0) convention.
# Reads:  FILES.CRS_PARQUET, FILES.PREDICTIONS_UNIQUE
# Writes: FILES.CRS_WITH_PREDICTIONS
# -----------------------------------------------------------------------------

#%% -----------------------------------------------------------------------------#
# Config & Paths
# -----------------------------------------------------------------------------#

import numpy as np
import pandas as pd

from utils import (
    load_config,
    read_parquet_pandas,
    resolve_path,
    safe_write_parquet_pandas,
)

PROJECT_ROOT, cfg = load_config(config_name="config.json")

PARQUET_COMPRESSION = cfg.get("PARQUET_COMPRESSION", "snappy")
SEED = int(cfg.get("SEED", 42))
SAMPLE_LIMIT = cfg.get("SAMPLE_LIMIT", None)  # optional quick test sampling

path_in_crs = resolve_path(PROJECT_ROOT, cfg["FILES"]["CRS_PARQUET"])
path_in_preds = resolve_path(PROJECT_ROOT, cfg["FILES"]["PREDICTIONS_UNIQUE"])
path_out = resolve_path(PROJECT_ROOT, cfg["FILES"]["CRS_WITH_PREDICTIONS"])

COL_TEXT = "ProjectDesc"
COL_KEY = "text_key"

PROB_COL = "prob_es"
PRED_COL = "pred_es"



#%% -----------------------------------------------------------------------------#
# Run Script
# -----------------------------------------------------------------------------#

df_crs = read_parquet_pandas(path_in_crs)
df_pred = read_parquet_pandas(path_in_preds)

assert COL_TEXT in df_crs.columns
assert COL_KEY in df_crs.columns

assert COL_TEXT in df_pred.columns
assert COL_KEY in df_pred.columns
assert PROB_COL in df_pred.columns
assert PRED_COL in df_pred.columns

# Predictions must be unique per text_key (full-pred is run on unique_projectdesc)
assert df_pred[COL_KEY].is_unique
assert df_pred[COL_TEXT].is_unique

df_pred_u = df_pred[[COL_KEY, COL_TEXT, PROB_COL, PRED_COL]].copy()

# Optional sampling for quick testing (sample from head)
if SAMPLE_LIMIT is not None:
    sample_n = int(SAMPLE_LIMIT)
    head_n = sample_n * 10000
    df_head = df_crs.head(head_n)
    df_crs = df_head.sample(n=sample_n, random_state=SEED).copy()

# Sanity check: merge on text_key equivalent to merge on ProjectDesc (on a small slice)
n_check = min(5000, len(df_crs))
df_check = df_crs.head(n_check).copy()

m_key = df_check.merge(
    df_pred_u[[COL_KEY, PROB_COL, PRED_COL]],
    on=COL_KEY,
    how="left",
)

m_txt = df_check.merge(
    df_pred_u[[COL_TEXT, PROB_COL, PRED_COL]],
    on=COL_TEXT,
    how="left",
    suffixes=("", "_txt"),
)

a_prob = m_key[PROB_COL].to_numpy()
a_pred = m_key[PRED_COL].to_numpy()
b_prob = m_txt[PROB_COL].to_numpy()
b_pred = m_txt[PRED_COL].to_numpy()

prob_equal = np.allclose(
    np.nan_to_num(a_prob, nan=-999.0),
    np.nan_to_num(b_prob, nan=-999.0),
    rtol=0.0,
    atol=0.0,
)
pred_equal = np.array_equal(
    np.nan_to_num(a_pred, nan=-999),
    np.nan_to_num(b_pred, nan=-999),
)

if not (prob_equal and pred_equal):
    diff_mask = (
        (np.nan_to_num(a_pred, nan=-999) != np.nan_to_num(b_pred, nan=-999))
        | (np.nan_to_num(a_prob, nan=-999.0) != np.nan_to_num(b_prob, nan=-999.0))
    )
    preview = (
        pd.DataFrame(
            {
                COL_KEY: m_key.loc[diff_mask, COL_KEY].head(5),
                COL_TEXT: m_key.loc[diff_mask, COL_TEXT].head(5),
                f"{PROB_COL}_key": m_key.loc[diff_mask, PROB_COL].head(5),
                f"{PROB_COL}_txt": m_txt.loc[diff_mask, f"{PROB_COL}_txt"].head(5),
                f"{PRED_COL}_key": m_key.loc[diff_mask, PRED_COL].head(5),
                f"{PRED_COL}_txt": m_txt.loc[diff_mask, f"{PRED_COL}_txt"].head(5),
            }
        )
        .reset_index(drop=True)
    )
    print("Sanity check FAILED: key-merge != text-merge (preview first 5 diffs):")
    print(preview.to_string(index=False))
    assert False

print(f"Sanity check OK: merge on text_key == merge on ProjectDesc (checked n={n_check:,})")

# Merge full CRS with step1 preds (text_key is the join contract)
df_out = df_crs.merge(
    df_pred_u[[COL_KEY, PROB_COL, PRED_COL]],
    on=COL_KEY,
    how="left",
)

# Fill unmatched as NO_ES convention
n_unmatched = int(df_out[PROB_COL].isna().sum())
print(f"Filled unmatched rows as NO_ES: n={n_unmatched:,}")
df_out[PROB_COL] = df_out[PROB_COL].astype("float64").fillna(0.0)
df_out[PRED_COL] = df_out[PRED_COL].astype("Int32").fillna(0).astype("int32")

safe_write_parquet_pandas(df_out, path_out, compression=PARQUET_COMPRESSION, index=False)

print(f"Wrote: {path_out} | Rows: {len(df_out):,} | Cols: {df_out.shape[1]:,}")
