"""Build the analytic ICU cohort from MIMIC-IV Demo v2.2.

Study design (retrospective cohort, prediction time = ICU intime + 24 h)
-----------------------------------------------------------------------
Inclusion / exclusion, applied in this order and logged to
``reports/cohort_flow.json``:

1. All ICU stays in ``icu/icustays``.
2. Keep the **first ICU stay per patient** (earliest ``intime``) so that every
   patient contributes exactly one independent observation.
3. Keep **adults** (age at ICU admission >= 18). MIMIC-IV shifts dates, so age is
   ``anchor_age + (year(intime) - anchor_year)``.
4. Exclude stays with **ICU LOS < 24 h**, or where the patient **died before
   the end of the 24 h observation window**. A model scored at hour 24 cannot be
   applied to these stays, and keeping them would leak the outcome into the
   features (very short stays are disproportionately early deaths).

Features (all observable by hour 24)
------------------------------------
* Demographics: age, sex, admission type (emergency / urgent), medical vs. surgical ICU.
* Labs: min / max / mean / count of creatinine, lactate, WBC, hemoglobin,
  sodium, potassium and platelets drawn in the window ``[intime, intime + 24h]``.
  Events are linked on ``subject_id`` + ``charttime`` so that labs missing a
  ``hadm_id`` (common for ED / pre-registration draws) are not silently dropped.
* Comorbidity flags from ICD-9/ICD-10 codes (see ``COMORBIDITY_CODES``).
  NOTE: MIMIC diagnosis codes are assigned at discharge. Chronic conditions
  (heart failure, CKD, diabetes, COPD, liver disease, cancer) are usually
  pre-existing and are the standard Elixhauser-style risk adjusters, but acute
  codes such as sepsis may partly reflect the ICU course. ``train.py`` therefore
  runs a sensitivity analysis without any ICD features.

Outcomes
--------
* ``mortality``      : in-hospital death (``admissions.hospital_expire_flag``).
* ``prolonged_los``  : ICU length of stay > 3 days.

Usage:
    python src/cohort.py [--raw data/raw] [--out data/processed]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"

OBS_WINDOW_HOURS = 24
PROLONGED_LOS_DAYS = 3.0
MIN_AGE = 18

# d_labitems itemids (blood only). Whole-blood / blood-gas variants are pooled
# with the chemistry / hematology analyte they measure.
LAB_ITEMS: dict[str, list[int]] = {
    "creatinine": [50912, 52024],
    "lactate": [50813],
    "wbc": [51301, 51300],
    "hemoglobin": [51222, 50811],
    "sodium": [50983, 50824],
    "potassium": [50971, 50822],
    "platelets": [51265],
}
LAB_AGGS = ("min", "max", "mean")

# ICD code prefixes (dots removed, as stored in MIMIC-IV).
COMORBIDITY_CODES: dict[str, dict[int, tuple[str, ...]]] = {
    "sepsis": {
        10: ("A40", "A41", "R652", "R6520", "R6521"),
        9: ("038", "99591", "99592", "78552"),
    },
    "renal_failure": {
        10: ("N17", "N18", "N19"),
        9: ("584", "585", "586"),
    },
    "heart_failure": {
        10: ("I50", "I110", "I130", "I132"),
        9: ("428", "40201", "40211", "40291", "40401", "40403", "40411",
            "40413", "40491", "40493"),
    },
    "diabetes": {10: ("E08", "E09", "E10", "E11", "E13"), 9: ("250",)},
    "copd": {10: ("J44",), 9: ("491", "492", "496")},
    "liver_disease": {10: ("K70", "K71", "K72", "K73", "K74", "K76"), 9: ("570", "571")},
    "malignancy": {10: tuple(f"C{i:02d}" for i in range(0, 97)), 9: tuple(str(i) for i in range(140, 209))},
}

ID_COLS = ["subject_id", "hadm_id", "stay_id"]
TARGETS = ["mortality", "prolonged_los"]

log = logging.getLogger("cohort")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    def rd(rel: str, **kw) -> pd.DataFrame:
        return pd.read_csv(raw_dir / rel, **kw)

    all_lab_ids = sorted({i for ids in LAB_ITEMS.values() for i in ids})
    labs = rd("hosp/labevents.csv.gz",
              usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
              parse_dates=["charttime"])
    labs = labs[labs["itemid"].isin(all_lab_ids)]
    return {
        "patients": rd("hosp/patients.csv.gz"),
        "admissions": rd("hosp/admissions.csv.gz", parse_dates=["admittime", "dischtime", "deathtime"]),
        "icustays": rd("icu/icustays.csv.gz", parse_dates=["intime", "outtime"]),
        "diagnoses": rd("hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str}),
        "labevents": labs,
        "d_labitems": rd("hosp/d_labitems.csv.gz"),
    }


# --------------------------------------------------------------------------- #
# Cohort selection
# --------------------------------------------------------------------------- #
def select_cohort(icustays: pd.DataFrame, patients: pd.DataFrame,
                  admissions: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    flow: list[dict] = []

    def step(name: str, df: pd.DataFrame) -> None:
        flow.append({"step": name, "n_stays": int(len(df)), "n_patients": int(df["subject_id"].nunique())})
        log.info("%-55s stays=%4d patients=%4d", name, len(df), df["subject_id"].nunique())

    df = icustays.copy()
    step("All ICU stays", df)

    df = df.sort_values(["subject_id", "intime", "stay_id"])
    df = df.groupby("subject_id", as_index=False).head(1)
    step("First ICU stay per patient", df)

    df = df.merge(patients[["subject_id", "gender", "anchor_age", "anchor_year"]], on="subject_id", how="left")
    df["age"] = df["anchor_age"] + (df["intime"].dt.year - df["anchor_year"])
    df = df[df["age"] >= MIN_AGE]
    step(f"Age >= {MIN_AGE} at ICU admission", df)

    df = df.merge(
        admissions[["hadm_id", "admittime", "dischtime", "deathtime", "admission_type",
                    "hospital_expire_flag"]],
        on="hadm_id", how="left")
    df["window_end"] = df["intime"] + pd.Timedelta(hours=OBS_WINDOW_HOURS)
    died_in_window = df["deathtime"].notna() & (df["deathtime"] <= df["window_end"])
    df = df[(df["outtime"] >= df["window_end"]) & ~died_in_window]
    step(f"ICU LOS >= {OBS_WINDOW_HOURS}h and alive at hour {OBS_WINDOW_HOURS}", df)

    return df.reset_index(drop=True), flow


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def extract_first24h_labs(labevents: pd.DataFrame, stays: pd.DataFrame,
                          hours: int = OBS_WINDOW_HOURS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate labs drawn in ``[intime, intime + hours]`` per stay.

    Returns ``(features, audit)`` where ``audit`` is the exact set of lab events
    used (one row per event, with intime/window_end) so leakage can be verified.
    """
    id_to_name = {i: name for name, ids in LAB_ITEMS.items() for i in ids}
    labs = labevents[labevents["itemid"].isin(id_to_name)].dropna(subset=["valuenum"]).copy()
    labs["lab"] = labs["itemid"].map(id_to_name)

    win = stays[["subject_id", "stay_id", "intime"]].copy()
    win["window_end"] = win["intime"] + pd.Timedelta(hours=hours)
    m = labs.merge(win, on="subject_id", how="inner")
    m = m[(m["charttime"] >= m["intime"]) & (m["charttime"] <= m["window_end"])]

    audit = m[["subject_id", "stay_id", "itemid", "lab", "charttime", "intime", "window_end", "valuenum"]]
    agg = m.groupby(["stay_id", "lab"])["valuenum"].agg([*LAB_AGGS, "count"]).unstack("lab")
    agg.columns = [f"{lab}_{stat}" for stat, lab in agg.columns]

    cols = [f"{lab}_{stat}" for lab in LAB_ITEMS for stat in (*LAB_AGGS, "count")]
    feats = agg.reindex(columns=cols).reindex(stays["stay_id"])
    count_cols = [c for c in cols if c.endswith("_count")]
    feats[count_cols] = feats[count_cols].fillna(0).astype(int)
    return feats.reset_index().rename(columns={"index": "stay_id"}), audit.reset_index(drop=True)


def comorbidity_flags(diagnoses: pd.DataFrame, hadm_ids: pd.Series) -> pd.DataFrame:
    dx = diagnoses[diagnoses["hadm_id"].isin(hadm_ids)].copy()
    dx["icd_code"] = dx["icd_code"].astype(str).str.replace(".", "", regex=False).str.strip().str.upper()
    out = pd.DataFrame({"hadm_id": hadm_ids.values})
    for name, by_ver in COMORBIDITY_CODES.items():
        hit = np.zeros(len(dx), dtype=bool)
        for ver, prefixes in by_ver.items():
            hit |= (dx["icd_version"] == ver).values & dx["icd_code"].str.startswith(prefixes).values
        positive = set(dx.loc[hit, "hadm_id"])
        out[f"dx_{name}"] = out["hadm_id"].isin(positive).astype(int)
    return out


def build_cohort(raw: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    stays, flow = select_cohort(raw["icustays"], raw["patients"], raw["admissions"])

    labs, audit = extract_first24h_labs(raw["labevents"], stays)
    dx = comorbidity_flags(raw["diagnoses"], stays["hadm_id"])

    df = stays.merge(labs, on="stay_id", how="left").merge(dx, on="hadm_id", how="left")
    df["male"] = (df["gender"] == "M").astype(int)
    df["emergency_admission"] = df["admission_type"].str.contains("EMER|URGENT", case=False, na=False).astype(int)
    # Medical-type units (MICU, MICU/SICU, CCU) vs. surgical / cardiac-surgical / neuro units.
    df["unit_medical"] = df["first_careunit"].str.contains(r"Medical|Coronary", na=False).astype(int)

    df["mortality"] = df["hospital_expire_flag"].astype(int)
    df["prolonged_los"] = (df["los"] > PROLONGED_LOS_DAYS).astype(int)

    keep = (ID_COLS + ["intime", "window_end", "first_careunit", "los"]
            + feature_columns(df) + TARGETS)
    return df[keep], audit, flow


def feature_columns(df: pd.DataFrame, include_icd: bool = True) -> list[str]:
    base = ["age", "male", "emergency_admission", "unit_medical"]
    labs = [f"{lab}_{s}" for lab in LAB_ITEMS for s in (*LAB_AGGS, "count")]
    icd = [f"dx_{n}" for n in COMORBIDITY_CODES] if include_icd else []
    return [c for c in base + labs + icd if c in df.columns]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the first-24h ICU cohort")
    p.add_argument("--raw", type=Path, default=RAW_DIR)
    p.add_argument("--out", type=Path, default=PROCESSED_DIR)
    p.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw = load_raw(args.raw)
    cohort, audit, flow = build_cohort(raw)

    args.out.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.out / "cohort.csv", index=False)
    audit.to_csv(args.out / "lab_events_used.csv", index=False)

    summary = {
        "flow": flow,
        "n_final": int(len(cohort)),
        "outcomes": {t: {"events": int(cohort[t].sum()), "prevalence": round(float(cohort[t].mean()), 4)}
                     for t in TARGETS},
        "lab_missingness_pct": {lab: round(float((cohort[f"{lab}_count"] == 0).mean() * 100), 1)
                                for lab in LAB_ITEMS},
        "comorbidity_prevalence_pct": {c: round(float(cohort[c].mean() * 100), 1)
                                       for c in cohort.columns if c.startswith("dx_")},
        "age_median_iqr": [float(x) for x in cohort["age"].quantile([0.5, 0.25, 0.75]).round(1)],
        "male_pct": round(float(cohort["male"].mean() * 100), 1),
    }
    (args.reports / "cohort_flow.json").write_text(json.dumps(summary, indent=2))
    log.info("Cohort: %d stays | %s", len(cohort), summary["outcomes"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
