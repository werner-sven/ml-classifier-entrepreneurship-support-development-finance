# -----------------------------------------------------------------------------
# 02_split_gold_data.py
# Read hand-labeled gold (xlsx), normalize ProjectDesc, create text_key, validate
# binary labels, stratified split into train/eval/test (shares from config).
# Reads:  FILES.GOLD_LABELED
# Writes: FILES.TRAIN_SPLIT, FILES.EVAL_SPLIT, FILES.TEST_SPLIT
# -----------------------------------------------------------------------------


#%% -----------------------------------------------------------------------------#
# Config & Paths
# -----------------------------------------------------------------------------#

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import (
    atomic_write_csv_pandas,
    load_config,
    resolve_path,
    norm_text,
    make_text_key,
)

PROJECT_ROOT, cfg = load_config(config_name="config.json")

SEED = int(cfg["SEED"])
CSV_SEP = str(cfg.get("CSV_SEP", "|"))

files = cfg["FILES"]
split_cfg = cfg["SPLIT"]

path_in_gold_xlsx = resolve_path(PROJECT_ROOT, files["GOLD_LABELED"])
path_out_train = resolve_path(PROJECT_ROOT, files["TRAIN_SPLIT"])
path_out_eval = resolve_path(PROJECT_ROOT, files["EVAL_SPLIT"])
path_out_test = resolve_path(PROJECT_ROOT, files["TEST_SPLIT"])

COL_TEXT = "ProjectDesc"
COL_Y = "gold_es"  


#%% -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#

def _validate_required_columns(df: pd.DataFrame) -> None:
    for c in (COL_TEXT, COL_Y):
        assert c in df.columns


def _validate_no_overlap(df_train: pd.DataFrame, df_eval: pd.DataFrame, df_test: pd.DataFrame) -> None:
    train_keys = set(df_train["text_key"].tolist())
    eval_keys = set(df_eval["text_key"].tolist())
    test_keys = set(df_test["text_key"].tolist())
    assert not ((train_keys & eval_keys) | (train_keys & test_keys) | (eval_keys & test_keys))


#%% -----------------------------------------------------------------------------#
# Run Script
# -----------------------------------------------------------------------------#

df = pd.read_excel(path_in_gold_xlsx, dtype=str, engine="openpyxl")
df.columns = [str(c).strip() for c in df.columns]

_validate_required_columns(df)

# Canonical normalization + stable key (same contract as CRS)
df[COL_TEXT] = df[COL_TEXT].map(norm_text)
df["text_key"] = df[COL_TEXT].map(make_text_key)

# Label -> numeric binary (fail loudly on junk)
df[COL_Y] = pd.to_numeric(df[COL_Y], errors="raise").astype("int32")
assert set(df[COL_Y].unique()).issubset({0, 1})

# Check if gold contains empty texts (after normalization)
assert (df[COL_TEXT] != "").all()

# Create splits (sklearn handles uneven sizes)
test_share = float(split_cfg["TEST_SHARE"])
eval_share_full = float(split_cfg["EVAL_SHARE"])
eval_share_of_remain = eval_share_full / (1.0 - test_share)

df_remain, df_test = train_test_split(
    df,
    test_size=test_share,
    random_state=SEED,
    stratify=df[COL_Y],
)

df_train, df_eval = train_test_split(
    df_remain,
    test_size=eval_share_of_remain,
    random_state=SEED,
    stratify=df_remain[COL_Y],
)

_validate_no_overlap(df_train, df_eval, df_test)

atomic_write_csv_pandas(df_train, path_out_train, sep=CSV_SEP)
atomic_write_csv_pandas(df_eval, path_out_eval, sep=CSV_SEP)
atomic_write_csv_pandas(df_test, path_out_test, sep=CSV_SEP)

print(f"[split] n={len(df)} -> train={len(df_train)} eval={len(df_eval)} test={len(df_test)}")
