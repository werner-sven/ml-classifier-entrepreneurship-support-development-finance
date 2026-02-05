# -----------------------------------------------------------------------------
# utils.py
# Shared minimal utilities: config + project root discovery, resolve_path, pandas
# parquet IO (read/write), atomic JSON/CSV writers, canonical norm_text + text_key,
# and timestamped run naming helpers.
# Used by: all scripts (00..06 + run_all)
# -----------------------------------------------------------------------------


#%% -----------------------------------------------------------------------------#
# Config & Paths
# -----------------------------------------------------------------------------#

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple


CONFIG_NAME_DEFAULT = "config.json"

def read_parquet_pandas(path: Path):
    import pandas as pd
    return pd.read_parquet(path)

def safe_write_parquet_pandas(df, path: Path, compression: str = "snappy", index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, compression=compression, index=index)
    os.replace(tmp, path)

def find_project_root(start: Path | None = None, config_name: str = CONFIG_NAME_DEFAULT) -> Path:
    """
    Walk upwards until we find the config file. This lets scripts run from anywhere.
    """
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / config_name).exists():
            return parent
    raise FileNotFoundError(f"Could not find {config_name} in any parent directory of {p}")

def load_config(config_name: str = CONFIG_NAME_DEFAULT) -> Tuple[Path, Dict[str, Any]]:
    project_root = find_project_root(config_name=config_name)
    cfg_path = project_root / config_name
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return project_root, cfg

def resolve_path(project_root: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (project_root / p)

# -----------------------------
# Time + run naming
# -----------------------------

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _sanitize_for_filename(s: str, max_len: int = 60) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return (s[:max_len] or "run").rstrip("-")

def make_stamp_name(description: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{_sanitize_for_filename(description)}"

# -----------------------------
# Text normalization + textkey
# -----------------------------

_WS_RE = re.compile(r"\s+", flags=re.UNICODE)

def norm_text(text: str) -> str:
    """
    One canonical normalization used for: split dedup, evaluation overlap, CRS merge robustness.
    Mirrors your split-script behavior (NBSP, whitespace collapse, strip). :contentReference[oaicite:11]{index=11}
    """
    if text is None:
        return ""
    s = str(text).replace("\u00A0", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s

def make_text_key(text: str) -> str:
    """
    Stable join key for normalized text.
    Helpful when merges should not fail on whitespace differences.
    """
    s = norm_text(text)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# -----------------------------
# Safe / atomic writers
# -----------------------------

def atomic_write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def atomic_write_csv_pandas(df, path: Path, sep: str = "|") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(
        tmp,
        index=False,
        sep=sep,
        quoting=csv.QUOTE_MINIMAL,
        quotechar='"',
        lineterminator="\n",
    )
    os.replace(tmp, path)