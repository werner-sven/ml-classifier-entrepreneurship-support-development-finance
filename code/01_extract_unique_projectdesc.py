# -----------------------------------------------------------------------------
# 01_extract_unique_projectdesc.py
# Build unique text table from CRS parquet: keep classifiable (ProjectDesc != ""),
# sanity-check text_key consistency, aggregate counts per (text_key, ProjectDesc).
# Reads:  FILES.CRS_PARQUET
# Writes: FILES.UNIQUE_PROJECTDESC
# -----------------------------------------------------------------------------


#%% -----------------------------------------------------------------------------#
# Config & Paths
# -----------------------------------------------------------------------------#

from utils import (
    load_config,
    resolve_path,
    read_parquet_pandas,
    safe_write_parquet_pandas,
    make_text_key,
)

PROJECT_ROOT, cfg = load_config(config_name="config.json")

path_in_crs = resolve_path(PROJECT_ROOT, cfg["FILES"]["CRS_PARQUET"])
path_out_unique = resolve_path(PROJECT_ROOT, cfg["FILES"]["UNIQUE_PROJECTDESC"])

PARQUET_COMPRESSION = cfg.get("PARQUET_COMPRESSION", "snappy")

COL_TEXT_NORM = "ProjectDesc"
COL_KEY = "text_key"


#%% -----------------------------------------------------------------------------#
# Run Script
# -----------------------------------------------------------------------------#

# CRS_RAW already contains normalized ProjectDesc + text_key (created in 00_ingest_crs.py)
df = read_parquet_pandas(path_in_crs)

# Classifiable criteria on normalized text:
# A row is classifiable iff ProjectDesc is not empty after normalization.
s = df[COL_TEXT_NORM].astype("string").fillna("")
is_classifiable = s != ""

# Sanity check: ProjectDesc <-> text_key equivalence (key should be hash(norm(ProjectDesc)))
# We recompute key from ProjectDesc and compare to stored text_key.
# Keep it simple and loud: if mismatch exists, we show a tiny preview and assert.
recomputed = df.loc[is_classifiable, COL_TEXT_NORM].map(make_text_key)
stored = df.loc[is_classifiable, COL_KEY].astype("string")
mismatch = stored.ne(recomputed)

if mismatch.any():
    preview = df.loc[is_classifiable, [COL_TEXT_NORM, COL_KEY]].loc[mismatch].head(5)
    print("Sanity check FAILED: ProjectDesc -> text_key mismatch. Preview (first 5):")
    print(preview.to_string(index=False))
    assert False

print("Sanity check OK: ProjectDesc and text_key are equivalent (hash of normalized text).")

df_unique = (
    df.loc[is_classifiable, [COL_TEXT_NORM, COL_KEY]]
    .groupby([COL_KEY, COL_TEXT_NORM], as_index=False)
    .size()
    .rename(columns={"size": "n_crs_rows"})
    .sort_values("n_crs_rows", ascending=False, kind="mergesort")
)

safe_write_parquet_pandas(df_unique, path_out_unique, compression=PARQUET_COMPRESSION, index=False)
print(f"Wrote: {path_out_unique} | Unique: {len(df_unique):,}")
