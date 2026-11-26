"""Leakage and inference tests for the first-24h ICU risk pipeline.

Each test has two layers:
* a deterministic unit test on small hand-built data (always runs), and
* a check on the real MIMIC-IV Demo artefacts produced by ``src/cohort.py`` /
  ``src/train.py`` (skipped only if the pipeline has not been run yet).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohort import OBS_WINDOW_HOURS, extract_first24h_labs, feature_columns, select_cohort  # noqa: E402
from train import cv_splits, make_pipeline  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
COHORT_CSV = PROCESSED / "cohort.csv"
AUDIT_CSV = PROCESSED / "lab_events_used.csv"
MODEL_PATH = ROOT / "models" / "model_pipeline.joblib"

needs_cohort = pytest.mark.skipif(not COHORT_CSV.exists(), reason="run src/cohort.py first")
needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run src/train.py first")


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return pd.read_csv(COHORT_CSV, parse_dates=["intime", "window_end"])


# --------------------------------------------------------------------------- #
# 1. No subject_id leakage between train and validation folds
# --------------------------------------------------------------------------- #
def _assert_disjoint(splits, subjects):
    assert splits, "no CV splits produced"
    seen_val = []
    for tr, va in splits:
        assert set(subjects[tr]).isdisjoint(subjects[va]), "subject_id in both train and validation"
        assert len(np.intersect1d(tr, va)) == 0
        seen_val.append(va)
    # every record is validated exactly once per repeat
    all_val = np.sort(np.concatenate(seen_val))
    assert np.array_equal(all_val, np.arange(len(subjects)))


def test_group_cv_no_subject_leakage_synthetic():
    rng = np.random.default_rng(0)
    # 60 patients with 1-4 stays each -> grouping actually matters here
    subjects = np.repeat(np.arange(60), rng.integers(1, 5, 60))
    y = np.repeat(rng.integers(0, 2, 60), 1)[subjects]
    for seed in range(5):
        _assert_disjoint(cv_splits(y, subjects, n_splits=5, seed=seed), subjects)


@needs_cohort
def test_group_cv_no_subject_leakage_real(cohort):
    subjects = cohort["subject_id"].to_numpy()
    assert cohort["subject_id"].is_unique, "first-ICU-stay restriction violated"
    for target in ("mortality", "prolonged_los"):
        y = cohort[target].to_numpy()
        for seed in range(10):
            _assert_disjoint(cv_splits(y, subjects, n_splits=5, seed=seed), subjects)


# --------------------------------------------------------------------------- #
# 2. Features come strictly from the first 24 h of the ICU stay
# --------------------------------------------------------------------------- #
def test_lab_window_excludes_pre_and_post_window_events():
    intime = pd.Timestamp("2150-01-01 08:00")
    stays = pd.DataFrame({"subject_id": [1], "stay_id": [10], "intime": [intime]})
    labs = pd.DataFrame({
        "subject_id": [1] * 5,
        "hadm_id": [100] * 5,
        "itemid": [50912] * 5,  # creatinine
        "charttime": [intime - pd.Timedelta(hours=2),      # before ICU -> excluded
                      intime,                                # boundary  -> included
                      intime + pd.Timedelta(hours=12),       # inside    -> included
                      intime + pd.Timedelta(hours=24),       # boundary  -> included
                      intime + pd.Timedelta(hours=30)],      # future    -> excluded
        "valuenum": [99.0, 1.0, 2.0, 3.0, 50.0],
    })
    feats, audit = extract_first24h_labs(labs, stays)
    row = feats.set_index("stay_id").loc[10]
    assert row["creatinine_count"] == 3
    assert row["creatinine_min"] == 1.0
    assert row["creatinine_max"] == 3.0
    assert row["creatinine_mean"] == pytest.approx(2.0)
    assert audit["valuenum"].tolist() == [1.0, 2.0, 3.0]


def test_cohort_excludes_stays_that_end_or_die_within_window():
    t0 = pd.Timestamp("2150-01-01")
    icu = pd.DataFrame({
        "subject_id": [1, 2, 3, 3], "hadm_id": [11, 21, 31, 32], "stay_id": [1, 2, 3, 4],
        "intime": [t0, t0, t0, t0 + pd.Timedelta(days=30)],
        "outtime": [t0 + pd.Timedelta(hours=10), t0 + pd.Timedelta(days=5),
                    t0 + pd.Timedelta(days=2), t0 + pd.Timedelta(days=33)],
        "los": [10 / 24, 5, 2, 3],
    })
    pats = pd.DataFrame({"subject_id": [1, 2, 3], "gender": ["M", "F", "M"],
                         "anchor_age": [60, 70, 17], "anchor_year": [2150] * 3})
    adm = pd.DataFrame({"hadm_id": [11, 21, 31, 32], "admittime": [t0] * 4, "dischtime": [t0] * 4,
                        "deathtime": [pd.NaT, t0 + pd.Timedelta(hours=20), pd.NaT, pd.NaT],
                        "admission_type": ["EW EMER."] * 4, "hospital_expire_flag": [0, 1, 0, 0]})
    out, _ = select_cohort(icu, pats, adm)
    # 1: LOS < 24h; 2: died at hour 20; 3: age 17 (and only first stay considered)
    assert out.empty


@needs_cohort
def test_real_lab_audit_within_first_24h(cohort):
    audit = pd.read_csv(AUDIT_CSV, parse_dates=["charttime", "intime", "window_end"])
    assert len(audit) > 0
    assert (audit["charttime"] >= audit["intime"]).all(), "lab drawn before ICU admission used"
    assert (audit["charttime"] <= audit["intime"] + pd.Timedelta(hours=OBS_WINDOW_HOURS)).all(), \
        "lab drawn after hour 24 used"
    assert set(audit["stay_id"]) <= set(cohort["stay_id"])
    # aggregated features must be exactly reproducible from the audited events only
    agg = audit.groupby(["stay_id", "lab"])["valuenum"].agg(["min", "max", "mean", "count"])
    merged = cohort.set_index("stay_id")
    for (stay, lab), r in agg.iterrows():
        assert merged.loc[stay, f"{lab}_min"] == pytest.approx(r["min"])
        assert merged.loc[stay, f"{lab}_max"] == pytest.approx(r["max"])
        assert merged.loc[stay, f"{lab}_mean"] == pytest.approx(r["mean"])
        assert merged.loc[stay, f"{lab}_count"] == r["count"]


@needs_cohort
def test_real_cohort_all_stays_reach_prediction_time(cohort):
    raw = pd.read_csv(ROOT / "data" / "raw" / "icu" / "icustays.csv.gz", parse_dates=["intime", "outtime"])
    m = cohort[["stay_id"]].merge(raw, on="stay_id")
    assert (m["outtime"] >= m["intime"] + pd.Timedelta(hours=OBS_WINDOW_HOURS)).all()
    assert (cohort["window_end"] == cohort["intime"] + pd.Timedelta(hours=OBS_WINDOW_HOURS)).all()


@needs_cohort
def test_no_outcome_or_post_window_columns_in_features(cohort):
    feats = feature_columns(cohort)
    forbidden = {"los", "outtime", "dischtime", "deathtime", "hospital_expire_flag",
                 "mortality", "prolonged_los", "discharge_location", "dod"}
    assert forbidden.isdisjoint(feats)


# --------------------------------------------------------------------------- #
# 3. Inference returns valid probabilities
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["logreg", "lightgbm"])
def test_pipeline_probabilities_synthetic(kind):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 6))
    X[rng.random(X.shape) < 0.2] = np.nan  # exercise imputation + indicators
    y = (rng.random(80) < 0.4).astype(int)
    p = make_pipeline(kind).fit(X, y).predict_proba(X)
    assert p.shape == (80, 2)
    assert np.all((p >= 0) & (p <= 1))
    assert np.allclose(p.sum(axis=1), 1)


@needs_model
@needs_cohort
def test_saved_model_probabilities_on_sample_records(cohort):
    bundle = joblib.load(MODEL_PATH)
    feats = bundle["features"]
    sample = cohort.sample(10, random_state=0)[feats].to_numpy(dtype=float)
    sample_missing = sample.copy()
    sample_missing[:, ::3] = np.nan  # a record with many missing labs must still score
    for target, entry in bundle["targets"].items():
        for X in (sample, sample_missing):
            p = entry["model"].predict_proba(X)[:, 1]
            assert p.shape == (10,)
            assert np.all(np.isfinite(p))
            assert np.all((p >= 0) & (p <= 1)), target
        assert 0 < entry["threshold"] < 1
