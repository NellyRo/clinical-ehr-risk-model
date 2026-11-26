"""Train and evaluate first-24h ICU risk models on the MIMIC-IV Demo cohort.

For each outcome (in-hospital mortality, ICU LOS > 3 days) this script compares

* **Logistic Regression** (L2-penalised, standardised inputs) - the baseline, and
* **LightGBM** (shallow, heavily regularised gradient-boosted trees),

inside one sklearn ``Pipeline`` each, so imputation and scaling are re-fitted
inside every training fold (no preprocessing leakage).

Missing data
------------
Labs are missing for a clinical reason (a lactate is only drawn when the
clinician suspects hypoperfusion), so missingness is informative. Every numeric
feature is median-imputed (median of the *training fold*) and a binary
``missingindicator_<feature>`` column is added for every feature that had
missing values in the training fold, letting both models learn from the fact
that a test was not ordered.

Validation
----------
Repeated stratified *group* K-fold (groups = ``subject_id``), so no patient ever
appears in both a training and a validation fold. Metrics are computed on the
pooled out-of-fold (OOF) predictions of each repeat (robust when a fold holds
only one or two events) and summarised as mean / SD across repeats, with a
95% percentile-bootstrap CI on the first repeat's OOF predictions.

Sensitivity / specificity use a threshold equal to the outcome prevalence of
the *training fold* - chosen without looking at validation labels.

Outputs: reports/metrics.json, reports/roc_pr_curves.png,
reports/feature_importance.png, reports/calibration.png,
models/model_pipeline.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort import PROCESSED_DIR, REPO_ROOT, TARGETS, feature_columns  # noqa: E402

REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"
SEED = 42

TARGET_LABELS = {"mortality": "In-hospital mortality", "prolonged_los": "ICU LOS > 3 days"}
MODEL_LABELS = {"logreg": "Logistic Regression", "lightgbm": "LightGBM"}
# Categorical slots 1-2 of the validated reference palette (dataviz skill).
COLORS = {"logreg": "#257de0", "lightgbm": "#f47230"}
INK, INK_2, GRID = "#191919", "#666666", "#e6e6e6"

warnings.filterwarnings("ignore", message="X does not have valid feature names")
log = logging.getLogger("train")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def make_pipeline(kind: str, seed: int = SEED) -> Pipeline:
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if kind == "logreg":
        return Pipeline([
            ("impute", imputer),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=5000)),
        ])
    if kind == "lightgbm":
        return Pipeline([
            ("impute", imputer),
            ("clf", LGBMClassifier(
                n_estimators=200, learning_rate=0.03, num_leaves=4, max_depth=2,
                min_child_samples=5, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.7, reg_lambda=1.0, random_state=seed,
                n_jobs=1, verbose=-1)),
        ])
    raise ValueError(f"Unknown model kind: {kind}")


def transformed_feature_names(pipe: Pipeline, features: list[str]) -> list[str]:
    imp: SimpleImputer = pipe.named_steps["impute"]
    names = list(features)
    if imp.indicator_ is not None:
        names += [f"missing_{features[i]}" for i in imp.indicator_.features_]
    return names


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def cv_splits(y: np.ndarray, groups: np.ndarray, n_splits: int = 5,
              seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified, patient-grouped folds: a subject_id never spans train & validation."""
    n_splits = int(min(n_splits, np.bincount(y.astype(int)).min()))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: np.ndarray) -> dict[str, float]:
    pred = p >= thr
    tp = np.sum(pred & (y == 1)); fn = np.sum(~pred & (y == 1))
    tn = np.sum(~pred & (y == 0)); fp = np.sum(pred & (y == 0))
    return {"sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1)}


def score(y: np.ndarray, p: np.ndarray, thr: np.ndarray) -> dict[str, float]:
    return {"auroc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), **threshold_metrics(y, p, thr)}


def bootstrap_ci(y: np.ndarray, p: np.ndarray, thr: np.ndarray, n_boot: int,
                 seed: int = SEED) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        for k, v in score(y[idx], p[idx], thr[idx]).items():
            draws.setdefault(k, []).append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def run_cv(df: pd.DataFrame, features: list[str], target: str, kind: str,
           n_splits: int, n_repeats: int, n_boot: int) -> dict:
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=int)
    groups = df["subject_id"].to_numpy()

    per_repeat: list[dict[str, float]] = []
    first_oof = first_thr = None
    for r in range(n_repeats):
        oof = np.full(len(y), np.nan)
        thr = np.full(len(y), np.nan)
        for tr, va in cv_splits(y, groups, n_splits, seed=SEED + r):
            pipe = make_pipeline(kind, seed=SEED + r).fit(X[tr], y[tr])
            oof[va] = pipe.predict_proba(X[va])[:, 1]
            thr[va] = y[tr].mean()
        per_repeat.append(score(y, oof, thr))
        if r == 0:
            first_oof, first_thr = oof, thr

    keys = per_repeat[0].keys()
    summary = {k: {"mean": float(np.mean([m[k] for m in per_repeat])),
                   "sd": float(np.std([m[k] for m in per_repeat], ddof=1)) if n_repeats > 1 else 0.0}
               for k in keys}
    ci = bootstrap_ci(y, first_oof, first_thr, n_boot)
    for k in summary:
        summary[k]["ci95_bootstrap"] = ci.get(k)
    return {"summary": summary, "oof": first_oof, "y": y}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_roc_pr(results: dict, path: Path) -> None:
    fig, axes = plt.subplots(len(results), 2, figsize=(11, 5 * len(results)), dpi=200)
    axes = np.atleast_2d(axes)
    for row, (target, res) in enumerate(results.items()):
        ax_roc, ax_pr = axes[row]
        prev = None
        for kind, r in res.items():
            y, p = r["y"], r["oof"]
            prev = y.mean()
            s = r["summary"]
            fpr, tpr, _ = roc_curve(y, p)
            ax_roc.plot(fpr, tpr, color=COLORS[kind], lw=2,
                        label=f"{MODEL_LABELS[kind]}  AUROC {s['auroc']['mean']:.3f}")
            prec, rec, _ = precision_recall_curve(y, p)
            ax_pr.step(rec, prec, where="post", color=COLORS[kind], lw=2,
                       label=f"{MODEL_LABELS[kind]}  AP {s['pr_auc']['mean']:.3f}")
        ax_roc.plot([0, 1], [0, 1], ls="--", color=INK_2, lw=1, label="Chance")
        ax_pr.axhline(prev, ls="--", color=INK_2, lw=1, label=f"Prevalence {prev:.3f}")
        n, ev = len(y), int(y.sum())
        ax_roc.set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.02))
        ax_pr.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", xlim=(0, 1), ylim=(0, 1.02))
        ax_roc.set_title(f"{TARGET_LABELS[target]} - ROC  (n={n}, events={ev})", color=INK, fontsize=11, loc="left")
        ax_pr.set_title(f"{TARGET_LABELS[target]} - Precision-Recall", color=INK, fontsize=11, loc="left")
        for ax in (ax_roc, ax_pr):
            _style(ax)
            ax.legend(frameon=False, fontsize=9, loc="lower right" if ax is ax_roc else "upper right")
    fig.suptitle("Out-of-fold curves (repeat 1 of repeated stratified group 5-fold CV); "
                 "legend values = mean over repeats", color=INK_2, fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_calibration(results: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 5), dpi=200)
    for ax, (target, res) in zip(np.atleast_1d(axes), results.items()):
        for kind, r in res.items():
            frac, mean_pred = calibration_curve(r["y"], r["oof"], n_bins=5, strategy="quantile")
            ax.plot(mean_pred, frac, marker="o", ms=7, lw=2, color=COLORS[kind],
                    label=f"{MODEL_LABELS[kind]}  Brier {r['summary']['brier']['mean']:.3f}")
        ax.plot([0, 1], [0, 1], ls="--", color=INK_2, lw=1, label="Perfect calibration")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed event rate", xlim=(0, 1), ylim=(0, 1))
        ax.set_title(f"{TARGET_LABELS[target]} - calibration (quintiles)", color=INK, fontsize=11, loc="left")
        _style(ax)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def importance_tables(final_models: dict, features: list[str]) -> dict:
    out = {}
    for target, models in final_models.items():
        out[target] = {}
        for kind, pipe in models.items():
            names = transformed_feature_names(pipe, features)
            clf = pipe.named_steps["clf"]
            if kind == "logreg":
                vals = clf.coef_.ravel()
            else:
                vals = clf.booster_.feature_importance(importance_type="gain")
                vals = vals / max(vals.sum(), 1e-12)
            out[target][kind] = pd.Series(vals, index=names[:len(vals)])
    return out


def plot_feature_importance(imps: dict, path: Path, top: int = 15) -> None:
    fig, axes = plt.subplots(len(imps), 2, figsize=(12, 5.5 * len(imps)), dpi=200)
    axes = np.atleast_2d(axes)
    for row, (target, by_kind) in enumerate(imps.items()):
        for col, kind in enumerate(("logreg", "lightgbm")):
            ax = axes[row, col]
            s = by_kind[kind]
            s = s.reindex(s.abs().sort_values(ascending=False).index)[:top][::-1]
            ax.barh(s.index, s.values, color=COLORS[kind], height=0.6, edgecolor="white", linewidth=1)
            if kind == "logreg":
                ax.axvline(0, color=INK_2, lw=1)
                ax.set_xlabel("Standardised coefficient (log-odds per SD)")
            else:
                ax.set_xlabel("Share of total split gain")
            ax.set_title(f"{TARGET_LABELS[target]} - {MODEL_LABELS[kind]}", color=INK, fontsize=11, loc="left")
            _style(ax)
            ax.grid(True, axis="x", color=GRID); ax.grid(False, axis="y")
            ax.tick_params(axis="y", labelsize=8.5, colors=INK)
    fig.suptitle(f"Top {top} features of the final models (refit on the full cohort)",
                 color=INK_2, fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _round(obj, nd: int = 4):
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), nd)
    return obj


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train & evaluate first-24h ICU risk models")
    p.add_argument("--cohort", type=Path, default=PROCESSED_DIR / "cohort.csv")
    p.add_argument("--reports", type=Path, default=REPORTS_DIR)
    p.add_argument("--models", type=Path, default=MODELS_DIR)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_csv(args.cohort)
    assert df["subject_id"].is_unique, "cohort must have one row per patient"
    features = feature_columns(df, include_icd=True)
    features_no_icd = feature_columns(df, include_icd=False)
    args.reports.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    sensitivity: dict[str, dict] = {}
    metrics: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "MIMIC-IV Clinical Database Demo v2.2 (PhysioNet, ODbL)",
        "prediction_time": "ICU intime + 24 h",
        "cohort": {"n_stays": int(len(df)), "n_patients": int(df["subject_id"].nunique()),
                   "n_features_raw": len(features)},
        "validation": {"scheme": "repeated StratifiedGroupKFold (groups=subject_id)",
                       "n_folds": args.folds, "n_repeats": args.repeats,
                       "metric_aggregation": "pooled out-of-fold per repeat; mean/sd across repeats",
                       "ci95": f"percentile bootstrap ({args.bootstrap} resamples) of repeat-1 OOF predictions",
                       "threshold": "training-fold outcome prevalence"},
        "targets": {},
    }

    for target in TARGETS:
        y = df[target]
        log.info("=== %s: %d events / %d (%.1f%%)", target, y.sum(), len(y), 100 * y.mean())
        results[target] = {}
        metrics["targets"][target] = {"label": TARGET_LABELS[target], "events": int(y.sum()),
                                      "prevalence": float(y.mean()), "models": {}}
        for kind in MODEL_LABELS:
            r = run_cv(df, features, target, kind, args.folds, args.repeats, args.bootstrap)
            results[target][kind] = r
            metrics["targets"][target]["models"][kind] = r["summary"]
            s = r["summary"]
            log.info("  %-20s AUROC %.3f±%.3f  PR-AUC %.3f  Brier %.3f  Sens %.3f  Spec %.3f",
                     MODEL_LABELS[kind], s["auroc"]["mean"], s["auroc"]["sd"], s["pr_auc"]["mean"],
                     s["brier"]["mean"], s["sensitivity"]["mean"], s["specificity"]["mean"])
            # Sensitivity analysis: drop discharge-coded ICD comorbidities.
            r2 = run_cv(df, features_no_icd, target, kind, args.folds, max(args.repeats // 2, 1), 0)
            sensitivity.setdefault(target, {})[kind] = {
                k: r2["summary"][k]["mean"] for k in ("auroc", "pr_auc", "brier")}

    metrics["sensitivity_analysis_no_icd_features"] = {
        "description": "Same CV without discharge-coded ICD comorbidity flags (labs + demographics only); "
                       f"{max(args.repeats // 2, 1)} repeats; mean of pooled-OOF metrics",
        "results": sensitivity,
    }

    # Final models: refit every candidate on the full cohort; select by mean CV AUROC.
    final_models: dict[str, dict[str, Pipeline]] = {}
    bundle_targets = {}
    X_all = df[features].to_numpy(dtype=float)
    for target in TARGETS:
        y = df[target].to_numpy(dtype=int)
        final_models[target] = {k: make_pipeline(k).fit(X_all, y) for k in MODEL_LABELS}
        aucs = {k: metrics["targets"][target]["models"][k]["auroc"]["mean"] for k in MODEL_LABELS}
        best = max(aucs, key=aucs.get)
        metrics["targets"][target]["selected_model"] = best
        bundle_targets[target] = {"model": final_models[target][best], "model_kind": best,
                                  "threshold": float(y.mean()),
                                  "cv_auroc_mean": aucs[best]}
        log.info("Selected %s for %s (CV AUROC %.3f)", best, target, aucs[best])

    bundle = {
        "features": features,
        "targets": bundle_targets,
        "all_models": final_models,
        "metadata": {"dataset": metrics["dataset"], "prediction_time": metrics["prediction_time"],
                     "trained_utc": metrics["generated_utc"], "n_train": int(len(df)),
                     "sklearn": sklearn.__version__, "python": platform.python_version()},
    }
    joblib.dump(bundle, args.models / "model_pipeline.joblib")

    imps = importance_tables(final_models, features)
    metrics["top_features"] = {t: {k: _round(s.reindex(s.abs().sort_values(ascending=False).index)[:10].to_dict())
                                   for k, s in by.items()} for t, by in imps.items()}

    plot_roc_pr(results, args.reports / "roc_pr_curves.png")
    plot_feature_importance(imps, args.reports / "feature_importance.png")
    plot_calibration(results, args.reports / "calibration.png")
    (args.reports / "metrics.json").write_text(json.dumps(_round(metrics), indent=2))
    log.info("Wrote metrics, plots and model bundle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
