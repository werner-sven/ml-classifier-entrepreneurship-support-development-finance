# -----------------------------------------------------------------------------
# 04_test_best_model.py
# Load best_model.json, predict on held-out test split, compute metrics + confusion
# matrix + qualitative TP/TN/FP/FN examples; save test artifacts + paper exports.
# Reads:  FILES.TEST_SPLIT, FILES.BEST_MODEL_JSON
# Writes: Best-Model/{test_metrics.json, confusion_matrix.png, examples_*.md, test_predictions_with_gold.parquet}
#         paper exports: FILES.BEST_CONFUSION_PNG, FILES.BEST_TEST_METRICS_CSV, FILES.BEST_TEST_METRICS_TEX
# -----------------------------------------------------------------------------


#%% -----------------------------------------------------------------------------#
# Config & Paths
# -----------------------------------------------------------------------------#

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import (
    atomic_write_csv_pandas,
    atomic_write_json,
    load_config,
    norm_text,
    now_iso_utc,
    resolve_path,
    safe_write_parquet_pandas,
)

PROJECT_ROOT, cfg = load_config(config_name="config.json")

CSV_SEP = cfg.get("CSV_SEP", "|")
PRED_BATCH_SIZE = int(cfg.get("PREDICT", {}).get("BATCH_SIZE", 64))

TEST_SPLIT_PATH = resolve_path(PROJECT_ROOT, cfg["FILES"]["TEST_SPLIT"])
BEST_MODEL_JSON_PATH = resolve_path(PROJECT_ROOT, cfg["FILES"]["BEST_MODEL_JSON"])

OUTPUT_BEST_DIR = resolve_path(PROJECT_ROOT, cfg["DIRS"]["OUTPUT_BEST"])
OUTPUT_REPORTS_DIR = resolve_path(PROJECT_ROOT, cfg["DIRS"]["OUTPUT_REPORTS"])

OUT_TEST_METRICS_JSON = OUTPUT_BEST_DIR / "test_metrics.json"
OUT_CM_PNG = OUTPUT_BEST_DIR / "confusion_matrix.png"
OUT_EXAMPLES_MD = OUTPUT_BEST_DIR / "examples_tp_tn_fp_fn.md"
OUT_TEST_PREDS_PARQUET = OUTPUT_BEST_DIR / "test_predictions_with_gold.parquet"

EXPORT_CM_PNG = resolve_path(PROJECT_ROOT, cfg["FILES"]["BEST_CONFUSION_PNG"])
EXPORT_METRICS_CSV = resolve_path(PROJECT_ROOT, cfg["FILES"]["BEST_TEST_METRICS_CSV"])
EXPORT_METRICS_TEX = resolve_path(PROJECT_ROOT, cfg["FILES"]["BEST_TEST_METRICS_TEX"])

TEXT_COL = "ProjectDesc"
LABEL_COL = "gold_es"  # 0/1


#%% -----------------------------------------------------------------------------#
# Helpers (lean)
# -----------------------------------------------------------------------------#

def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

def _snip(s: str, n: int = 500) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else (s[:n] + " ...")

def _softmax_pos_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]


#%% -----------------------------------------------------------------------------#
# Run Script
# -----------------------------------------------------------------------------#

with BEST_MODEL_JSON_PATH.open("r", encoding="utf-8") as f:
    best = json.load(f)

thr = float(best.get("threshold_eval") or best.get("best_thr") or best.get("best_thr_eval"))

best_model_dir = Path(best["paths"]["best_model_dir"])
best_model_dir = best_model_dir if best_model_dir.is_absolute() else resolve_path(PROJECT_ROOT, best_model_dir)

max_length = int(best.get("hparams", {}).get("max_length") or cfg.get("TRAIN", {}).get("MAX_LENGTH", 256))

df_raw = pd.read_csv(TEST_SPLIT_PATH, sep=CSV_SEP, dtype=str)
assert TEXT_COL in df_raw.columns
assert LABEL_COL in df_raw.columns

df = df_raw[[TEXT_COL, LABEL_COL]].copy()
df["text"] = df[TEXT_COL].astype(str).map(norm_text)
df["labels"] = pd.to_numeric(df[LABEL_COL], errors="raise").astype(int).clip(0, 1)
assert (df["text"] != "").all()

tokenizer = AutoTokenizer.from_pretrained(best_model_dir, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

texts = df["text"].tolist()
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

y_true = df["labels"].to_numpy(dtype=int)
y_hat = (probs >= thr).astype(int)

acc = float(accuracy_score(y_true, y_hat))
prec = float(precision_score(y_true, y_hat, zero_division=0))
rec = float(recall_score(y_true, y_hat, zero_division=0))
f1 = float(f1_score(y_true, y_hat, zero_division=0))

cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
tn, fp, fn, tp = [int(x) for x in cm.ravel()]

OUTPUT_BEST_DIR.mkdir(parents=True, exist_ok=True)

payload = {
    "created_utc": now_iso_utc(),
    "run_id": best.get("run_id"),
    "model_name": best.get("model_name"),
    "selection_metric": best.get("selection_metric"),
    "selection_value": best.get("selection_value"),
    "threshold_eval": thr,
    "test_size": int(len(df)),
    "metrics_at_threshold": {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    },
    "confusion_matrix": {
        "labels": ["NO_ES", "ES"],
        "matrix": cm.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    },
    "paths": {
        "best_model_dir": str(best_model_dir),
        "test_split": str(TEST_SPLIT_PATH),
    },
}
atomic_write_json(payload, OUT_TEST_METRICS_JSON)

df_out = df_raw.copy()
df_out[TEXT_COL] = df["text"]
df_out["prob_ES"] = probs.astype(float)
df_out["pred_ES_thr"] = y_hat.astype(int)
df_out["pred_label"] = np.where(df_out["pred_ES_thr"] == 1, "ES", "NO_ES")
df_out["gold_ES"] = y_true.astype(int)
df_out["pred_correct"] = (df_out["pred_ES_thr"].to_numpy() == df_out["gold_ES"].to_numpy())

safe_write_parquet_pandas(df_out, OUT_TEST_PREDS_PARQUET, compression=cfg.get("PARQUET_COMPRESSION", "snappy"), index=False)

# Confusion matrix figure
import matplotlib.pyplot as plt  # local import keeps top lean

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NO_ES", "ES"])
fig, ax = plt.subplots(figsize=(4.2, 4.2))
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title(f"Confusion Matrix (Best @ thr={thr:.2f})")
fig.tight_layout()
fig.savefig(OUT_CM_PNG, dpi=220)
plt.close(fig)

EXPORT_CM_PNG.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(OUT_CM_PNG, EXPORT_CM_PNG)

# Examples (5 each)
tp_mask = (df_out["gold_ES"] == 1) & (df_out["pred_ES_thr"] == 1)
tn_mask = (df_out["gold_ES"] == 0) & (df_out["pred_ES_thr"] == 0)
fp_mask = (df_out["gold_ES"] == 0) & (df_out["pred_ES_thr"] == 1)
fn_mask = (df_out["gold_ES"] == 1) & (df_out["pred_ES_thr"] == 0)

tp_ex = df_out.loc[tp_mask].sort_values("prob_ES", ascending=False).head(5)
tn_ex = df_out.loc[tn_mask].sort_values("prob_ES", ascending=True).head(5)
fp_ex = df_out.loc[fp_mask].sort_values("prob_ES", ascending=False).head(5)
fn_ex = df_out.loc[fn_mask].sort_values("prob_ES", ascending=True).head(5)

def _examples_block(title: str, ex: pd.DataFrame) -> str:
    lines = [f"## {title} (showing {len(ex)})", ""]
    for _, r in ex.iterrows():
        lines.append(f"- prob_ES={float(r['prob_ES']):.3f} | gold={int(r['gold_ES'])} | pred={int(r['pred_ES_thr'])}")
        lines.append(f"  - text: {_snip(str(r[TEXT_COL]))}")
    lines.append("")
    return "\n".join(lines)

md = []
md.append("# Best model on TEST — Examples (TP/TN/FP/FN)")
md.append("")
md.append(f"- run_id: {best.get('run_id')}")
md.append(f"- model_name: {best.get('model_name')}")
md.append(f"- threshold_eval: {thr:.3f}")
md.append("")

md.append(_examples_block("True Positives", tp_ex))
md.append(_examples_block("True Negatives", tn_ex))
md.append(_examples_block("False Positives", fp_ex))
md.append(_examples_block("False Negatives", fn_ex))

_atomic_write_text("\n".join(md).strip() + "\n", OUT_EXAMPLES_MD)

# Paper export table (single row) — LaTeX style aligned with comparison table script
df_metrics = pd.DataFrame(
    [
        {
            "run_id": best.get("run_id"),
            "model_name": best.get("model_name"),
            "threshold_eval": thr,
            "test_size": int(len(df)),
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
    ]
)

atomic_write_csv_pandas(df_metrics, EXPORT_METRICS_CSV, sep=CSV_SEP)

# LaTeX (booktabs + threeparttable), single-model version
model_name_tex = str(best.get("model_name") or "Best model").replace("_", r"\_")
n_test = int(len(df))
es_prev_pct = float(y_true.mean() * 100.0)


latex = rf"""% Requires in preamble:
% \usepackage{{booktabs}}
% \usepackage{{threeparttable}}

\begin{{table}}[!htbp]
  \centering
  \begingroup
    \footnotesize
    \setlength{{\tabcolsep}}{{10pt}}        % more horizontal space between columns
    \renewcommand{{\arraystretch}}{{1.15}}  % more vertical space

    \begin{{threeparttable}}
      \caption{{Performance of machine learning classifier on held-out test set for predicting ESDF}}
      \label{{tab:best_model_test}}

      \begin{{tabular}}{{llc}}
        \toprule
        Metric & Explanation & Value \\
        \midrule
        Accuracy        & Share of all predictions that are correct & {acc:.2f}  \\
        F1 (ESDF)       & Balance of precision and recall for ESDF  & {f1:.2f}   \\
        Precision (ESDF)& Share of predicted ESDF that are truly ESDF & {prec:.2f} \\
        Recall (ESDF)   & Share of true ESDF correctly predicted   & {rec:.2f}  \\
        \midrule
        Test set size ($N$)   & Size of held-out test set & {n_test:d} \\
        ESDF test prevalence (\%)    & Share of ESDF in the test data & {es_prev_pct:.0f} \\
        \bottomrule
      \end{{tabular}}

      \begin{{tablenotes}}[para,flushleft]
        \footnotesize
        \item \textit{{Note:}} 
        Pretrained language model finetuned on study data. Model was trained on the hand-labeld training set, tuned on the evaluation set, and performance metrics were calculated once on the untouched test set.
      \end{{tablenotes}}
    \end{{threeparttable}}
  \endgroup
\end{{table}}
""".strip() + "\n"

_atomic_write_text(latex, EXPORT_METRICS_TEX)

print(f"[test] acc={acc:.3f} f1={f1:.3f} p={prec:.3f} r={rec:.3f} thr={thr:.2f} | n={len(df)}")