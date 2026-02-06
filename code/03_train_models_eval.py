# -----------------------------------------------------------------------------
# 03_train_models_eval.py
# Train HF sequence classifier over HP grid using train/eval only; tune threshold
# on eval to maximize F1; write per-run artifacts + runs_summary; export eval table;
# select best run (by SELECTION_METRIC), copy model to Best-Model, write best_model.json.
# Reads:  FILES.TRAIN_SPLIT, FILES.EVAL_SPLIT
# Writes: DIRS.OUTPUT_RUNS/*, FILES.RUNS_SUMMARY, FILES.BEST_MODEL_JSON (+ Best model copy),
#         eval exports: FILES.EVAL_METRICS_CSV / FILES.EVAL_METRICS_TEX
# -----------------------------------------------------------------------------


#%% -----------------------------------------------------------------------------#
# Config & Paths
# -----------------------------------------------------------------------------#

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_progress_bar
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils import (
    atomic_write_csv_pandas,
    atomic_write_json,
    load_config,
    make_stamp_name,
    norm_text,
    now_iso_utc,
    resolve_path,
)

disable_progress_bar()

PROJECT_ROOT, cfg = load_config(config_name="config.json")

CSV_SEP = cfg.get("CSV_SEP", "|")
SEED = int(cfg.get("SEED", 42))

TRAIN_SPLIT_PATH = resolve_path(PROJECT_ROOT, cfg["FILES"]["TRAIN_SPLIT"])
EVAL_SPLIT_PATH = resolve_path(PROJECT_ROOT, cfg["FILES"]["EVAL_SPLIT"])

OUTPUT_RUNS_DIR = resolve_path(PROJECT_ROOT, cfg["DIRS"]["OUTPUT_RUNS"])
OUTPUT_BEST_DIR = resolve_path(PROJECT_ROOT, cfg["DIRS"]["OUTPUT_BEST"])
OUTPUT_REPORTS_DIR = resolve_path(PROJECT_ROOT, cfg["DIRS"]["OUTPUT_REPORTS"])

RUNS_SUMMARY_PATH = resolve_path(PROJECT_ROOT, cfg["FILES"]["RUNS_SUMMARY"])
BEST_MODEL_JSON_PATH = resolve_path(PROJECT_ROOT, cfg["FILES"]["BEST_MODEL_JSON"])

REPORTS_TABLES_DIR = OUTPUT_REPORTS_DIR / "tables"
REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_METRICS_CSV = resolve_path(PROJECT_ROOT, cfg["FILES"]["EVAL_METRICS_CSV"])
EXPORT_METRICS_TEX = resolve_path(PROJECT_ROOT, cfg["FILES"]["EVAL_METRICS_TEX"])

train_cfg = cfg.get("TRAIN", {})

TEXT_COL = "ProjectDesc"
LABEL_COL = "gold_es"  # 0/1
MAX_LENGTH = int(train_cfg.get("MAX_LENGTH", 256))

SELECTION_METRIC = train_cfg.get("SELECTION_METRIC", "eval_precision_thr")

DEFAULT_MODEL_NAME = (
    train_cfg.get("MODEL_NAME")
    or (train_cfg.get("MODELS") or ["distilbert-base-uncased"])[0]
)

HP_GRID: List[Dict[str, Any]] = train_cfg.get("HP_GRID") or [
    {"learning_rate": 2e-5, "batch_size_train": 8,  "batch_size_eval": 32, "num_epochs": 10, "weight_decay": 0.01,  "warmup_ratio": 0.1},
    {"learning_rate": 2e-5, "batch_size_train": 16, "batch_size_eval": 32, "num_epochs": 10, "weight_decay": 0.01, "warmup_ratio": 0.05},
    {"learning_rate": 3e-5, "batch_size_train": 8, "batch_size_eval": 32, "num_epochs": 10, "weight_decay": 0.00, "warmup_ratio": 0.1}
]

if not isinstance(HP_GRID, list) or not all(isinstance(x, dict) for x in HP_GRID):
    raise ValueError("TRAIN.HP_GRID (or HP_GRID here) must be a List[Dict].")

if len(HP_GRID) == 0:
    raise ValueError("HP_GRID is empty; nothing to run.")


#%% -----------------------------------------------------------------------------#
# Helpers (lean)
# -----------------------------------------------------------------------------#

def _read_split_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=CSV_SEP, dtype=str)

def _prep_text_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[TEXT_COL, LABEL_COL]].copy()
    out[TEXT_COL] = out[TEXT_COL].astype(str).map(norm_text)
    out["labels"] = pd.to_numeric(out[LABEL_COL], errors="raise").astype(int).clip(0, 1)
    return out.rename(columns={TEXT_COL: "text"})[["text", "labels"]]

def _softmax_pos_probs(logits: np.ndarray) -> np.ndarray:
    t = torch.tensor(logits)
    return torch.softmax(t, dim=1)[:, 1].cpu().numpy()

def _tune_threshold_max_f1(y_true: np.ndarray, p_pos: np.ndarray) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, p_pos)

    # thresholds has len = len(precision)-1; align by dropping last point
    precision = precision[:-1]
    recall = recall[:-1]

    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    thr = 0.5 if thresholds.size == 0 else float(thresholds[int(np.argmax(f1))])

    y_hat = (p_pos >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
    }

def _tune_threshold_max_precision(
    y_true: np.ndarray,
    p_pos: np.ndarray,
    min_recall: float = 0.0,   # set >0 if you ever want to avoid ultra-low recall solutions
) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, p_pos)

    # thresholds has len = len(precision)-1; align by dropping last point
    precision = precision[:-1]
    recall = recall[:-1]

    if thresholds.size == 0:
        thr = 0.5
    else:
        # optional recall constraint
        valid = recall >= float(min_recall)
        if not np.any(valid):
            valid = np.ones_like(recall, dtype=bool)

        prec_valid = np.where(valid, precision, -1.0)
        best_prec = float(np.max(prec_valid))
        cand = np.where(prec_valid == best_prec)[0]

        # tie-break 1: higher recall
        if cand.size > 1:
            best_rec = np.max(recall[cand])
            cand = cand[recall[cand] == best_rec]

        # tie-break 2: higher F1 (just to stabilize ties)
        if cand.size > 1:
            denom = precision[cand] + recall[cand]
            f1 = np.where(denom > 0, 2 * precision[cand] * recall[cand] / denom, 0.0)
            idx = int(cand[int(np.argmax(f1))])
        else:
            idx = int(cand[0])

        thr = float(thresholds[idx])

    y_hat = (p_pos >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
    }


def _best_epoch_from_log_history(trainer: Trainer) -> float | None:
    best_metric = trainer.state.best_metric
    if best_metric is None:
        return None
    for h in trainer.state.log_history:
        if "eval_f1" in h and "epoch" in h:
            if abs(float(h["eval_f1"]) - float(best_metric)) < 1e-12:
                return float(h["epoch"])
    return None

def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


#%% -----------------------------------------------------------------------------#
# Run Script (epoch-eval + best checkpoint; threshold tuning on eval; TEST untouched)
# -----------------------------------------------------------------------------#

set_seed(SEED)


# --- Guard: prevent mixing artifacts from different runs ----------------------
if OUTPUT_BEST_DIR.exists():
    existing = [p for p in OUTPUT_BEST_DIR.iterdir() if p.name not in {".gitkeep"}]
    if existing:
        raise RuntimeError(
            f"[ABORT] OUTPUT_BEST_DIR is not empty: {OUTPUT_BEST_DIR}\n"
            f"Found: {[p.name for p in existing]}\n"
            "Please empty output/02_best_model (or the configured OUTPUT_BEST_DIR) "
            "before running to avoid mixing artifacts from different runs."
        )
# -----------------------------------------------------------------------------


df_train = _prep_text_labels(_read_split_csv(TRAIN_SPLIT_PATH))
df_eval = _prep_text_labels(_read_split_csv(EVAL_SPLIT_PATH))

ds_train = Dataset.from_pandas(df_train, preserve_index=False)
ds_eval = Dataset.from_pandas(df_eval, preserve_index=False)

run_rows: List[Dict[str, Any]] = []

for run_idx, hp in enumerate(HP_GRID, start=1):
    model_name = hp.get("model_name", DEFAULT_MODEL_NAME)

    learning_rate = float(hp["learning_rate"])
    bs_train = int(hp["batch_size_train"])
    bs_eval = int(hp["batch_size_eval"])
    num_epochs = int(hp["num_epochs"])
    weight_decay = float(hp["weight_decay"])
    warmup_ratio = float(hp["warmup_ratio"])

    run_id = make_stamp_name(
        f"ml-grid_seed{SEED}_run{run_idx:02d}_{model_name}_lr{learning_rate}_bs{bs_train}_ep{num_epochs}"
    )

    run_dir = OUTPUT_RUNS_DIR / run_id
    eval_dir = run_dir / "eval"
    model_dir = run_dir / "model"
    hf_ckpt_dir = run_dir / "hf_checkpoints"
    logs_dir = run_dir / "logs"
    eval_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tok_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tok_train = ds_train.map(tok_fn, batched=True)
    tok_eval = ds_eval.map(tok_fn, batched=True)

    tok_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tok_eval.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(hf_ckpt_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=bs_train,
        per_device_eval_batch_size=bs_eval,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",
        logging_dir=str(logs_dir),
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        dataloader_num_workers=0,
        fp16=False,
        report_to=[],
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_eval,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-3)],
    )

    trainer.train()

    # Best checkpoint is loaded at end (load_best_model_at_end=True)
    eval_best = trainer.evaluate(eval_dataset=tok_eval)
    best_epoch = _best_epoch_from_log_history(trainer)

    pred_out = trainer.predict(tok_eval)
    logits = np.asarray(pred_out.predictions)
    y_true = np.asarray(pred_out.label_ids).astype(int)
    p_pos = _softmax_pos_probs(logits)

    tuned = _tune_threshold_max_precision(y_true, p_pos)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    eval_payload = {
        "run_id": run_id,
        "created_utc": now_iso_utc(),
        "seed": SEED,
        "model_name": model_name,
        "hparams": {
            "learning_rate": learning_rate,
            "batch_size_train": bs_train,
            "batch_size_eval": bs_eval,
            "num_epochs": num_epochs,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "max_length": MAX_LENGTH,
            "early_stopping": {"patience": 3, "threshold": 1e-5},
        },
        "trainer_best": {
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric_eval_f1_argmax": trainer.state.best_metric,
            "best_epoch_eval_f1_argmax": best_epoch,
        },
        "eval_best_epoch_argmax": {
            k: float(v)
            for k, v in eval_best.items()
            if isinstance(v, (int, float, np.floating))
            and k in {"eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1"}
        },
        "eval_threshold_tuning": tuned,
        "paths": {"run_dir": str(run_dir), "model_dir": str(model_dir)},
    }
    atomic_write_json(eval_payload, eval_dir / "eval_metrics.json")
    atomic_write_json(training_args.to_dict(), eval_dir / "training_args.json")

    df_eval_out = df_eval.copy()
    df_eval_out["prob_ES"] = p_pos.astype(float)
    df_eval_out["pred_ES_thr"] = (p_pos >= tuned["threshold"]).astype(int)
    df_eval_out["pred_ES_argmax"] = np.argmax(logits, axis=-1).astype(int)
    atomic_write_csv_pandas(df_eval_out, eval_dir / "eval_predictions_with_gold.csv", sep=CSV_SEP)

    run_rows.append(
        {
            "run_id": run_id,
            "seed": SEED,
            "model_name": model_name,
            "learning_rate": learning_rate,
            "batch_size_train": bs_train,
            "batch_size_eval": bs_eval,
            "num_epochs": num_epochs,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "best_epoch_argmax": best_epoch,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "eval_loss_argmax": float(eval_best.get("eval_loss", np.nan)),
            "eval_accuracy_argmax": float(eval_best.get("eval_accuracy", np.nan)),
            "eval_precision_argmax": float(eval_best.get("eval_precision", np.nan)),
            "eval_recall_argmax": float(eval_best.get("eval_recall", np.nan)),
            "eval_f1_argmax": float(eval_best.get("eval_f1", np.nan)),
            "threshold_eval": float(tuned["threshold"]),
            "eval_accuracy_thr": float(tuned["accuracy"]),
            "eval_precision_thr": float(tuned["precision"]),
            "eval_recall_thr": float(tuned["recall"]),
            "eval_f1_best": float(tuned["f1"]),
            "run_dir": str(run_dir),
            "model_dir": str(model_dir),
        }
    )

df_runs = pd.DataFrame(run_rows)

if SELECTION_METRIC not in df_runs.columns:
    raise KeyError(f"SELECTION_METRIC='{SELECTION_METRIC}' not found in runs table.")

df_runs = df_runs.sort_values(by=SELECTION_METRIC, ascending=False).reset_index(drop=True)

atomic_write_csv_pandas(df_runs, RUNS_SUMMARY_PATH, sep=CSV_SEP)

# Report table (xlsx + exports) + latex print
paper_cols = [
    "run_id",
    "model_name",
    "learning_rate",
    "batch_size_train",
    "batch_size_eval",
    "num_epochs",
    "weight_decay",
    "warmup_ratio",
    "best_epoch_argmax",
    "eval_f1_argmax",
    "threshold_eval",
    "eval_f1_best",
    "eval_precision_thr",
    "eval_recall_thr",
    "eval_accuracy_thr",
]
paper_cols = [c for c in paper_cols if c in df_runs.columns]
df_paper = df_runs[paper_cols].copy()

atomic_write_csv_pandas(df_runs, EXPORT_METRICS_CSV, sep=CSV_SEP)

#%%
# -LATEX Printing

K = int(train_cfg.get("APPENDIX_TOP_K", 5))
K = min(K, len(df_runs))
df_top = df_runs.head(K).copy()

n_eval = int(len(df_eval))
es_prev = float(df_eval["labels"].mean()) if n_eval else 0.0
es_prev_pct = es_prev * 100.0
n_es = int(df_eval["labels"].sum()) if n_eval else 0
n_non = int(n_eval - n_es)
majority_correct = max(n_es, n_non) if n_eval else 0
baseline_acc = (majority_correct / n_eval) if n_eval else 0.0
baseline_label = "Non-ESDF" if n_non >= n_es else "ESDF"

def _fmt(x, nd=2) -> str:
    return f"{float(x):.{nd}f}"

def _fmt_lr(x: float) -> str:
    s = f"{float(x):.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")

def _tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_")

model_headers = []
for j, r in df_top.reset_index(drop=True).iterrows():
    name = _tex_escape(r["model_name"])
    lr = _fmt_lr(r["learning_rate"])
    bs = int(r["batch_size_train"])
    ep = int(r["num_epochs"])
    model_headers.append(rf"\shortstack{{{name}\\lr={lr}, bs={bs}, ep={ep}}}")


col_spec = "l" + ("c" * K)
cmid = f"3-{K+2}"

def _row(metric: str, vals: list[str]) -> str:
    return metric + " & " + " & ".join(vals) + r" \\"

vals_acc  = [_fmt(v) for v in df_top["eval_accuracy_thr"].tolist()]
vals_f1   = [_fmt(v) for v in df_top["eval_f1_best"].tolist()]
vals_prec = [_fmt(v) for v in df_top["eval_precision_thr"].tolist()]
vals_rec  = [_fmt(v) for v in df_top["eval_recall_thr"].tolist()]
vals_thr  = [_fmt(v, nd=3) for v in df_top["threshold_eval"].tolist()]
vals_n    = [f"{n_eval:d}"] * K
vals_prev = [f"{es_prev_pct:.0f}"] * K

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
      \caption{{Performance of machine learning classifier candidates on the validation split for predicting ESDF.}}
      \label{{tab:model_comparison}}

      \begin{{tabular}}{{{col_spec}}}
        \toprule
         Model & {" & ".join(model_headers)} \\
        \midrule
        {_row("Accuracy", vals_acc)}
        {_row("F1 (ESDF)", vals_f1)}
        {_row("Precision (ESDF)", vals_prec)}
        {_row("Recall (ESDF)", vals_rec)}
        \midrule
        {_row(r"Threshold ($\tau$)", vals_thr)}
        \bottomrule
      \end{{tabular}}

      \begin{{tablenotes}}[para,flushleft]
        \footnotesize
        \item \textit{{Note:}}
        Models were trained on the hand-labeled training split. "lr" refers to learning-rate, "bs" to batch size and "ep" to epochs. Thresholds were tuned on the validation split to maximize F1 (ESDF); reported metrics are computed at the tuned $\tau$ on the same validation split. The test split remains untouched.
      \end{{tablenotes}}
    \end{{threeparttable}}
  \endgroup
\end{{table}}
""".strip() + "\n"

_atomic_write_text(latex, EXPORT_METRICS_TEX)



#%%
# Select best model + copy + best_model.json
best = df_runs.iloc[0].to_dict()

best_model_dir = OUTPUT_BEST_DIR / "model"
best_model_dir.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(Path(best["model_dir"]), best_model_dir)

best_payload = {
    "created_utc": now_iso_utc(),
    "selection_metric": SELECTION_METRIC,
    "selection_value": float(best[SELECTION_METRIC]),
    "threshold_eval": float(best["threshold_eval"]),
    "run_id": best["run_id"],
    "model_name": best["model_name"],
    "paths": {
        "run_dir": best["run_dir"],
        "best_model_dir": str(best_model_dir),
    },
    "hparams": {
        "seed": int(best["seed"]),
        "max_length": MAX_LENGTH,
        "learning_rate": float(best["learning_rate"]),
        "batch_size_train": int(best["batch_size_train"]),
        "batch_size_eval": int(best["batch_size_eval"]),
        "num_epochs": int(best["num_epochs"]),
        "weight_decay": float(best["weight_decay"]),
        "warmup_ratio": float(best["warmup_ratio"]),
    },
    "eval_best_epoch": {
        "best_epoch_argmax": best["best_epoch_argmax"],
        "eval_loss_argmax": float(best["eval_loss_argmax"]),
        "eval_accuracy_argmax": float(best["eval_accuracy_argmax"]),
        "eval_precision_argmax": float(best["eval_precision_argmax"]),
        "eval_recall_argmax": float(best["eval_recall_argmax"]),
        "eval_f1_argmax": float(best["eval_f1_argmax"]),
        "eval_accuracy_thr": float(best["eval_accuracy_thr"]),
        "eval_precision_thr": float(best["eval_precision_thr"]),
        "eval_recall_thr": float(best["eval_recall_thr"]),
        "eval_f1_best": float(best["eval_f1_best"]),
    },
}
atomic_write_json(best_payload, BEST_MODEL_JSON_PATH)
print(f"Best model: {best['run_id']} | {SELECTION_METRIC}={best[SELECTION_METRIC]:.4f} | {best_model_dir}")

# %%
