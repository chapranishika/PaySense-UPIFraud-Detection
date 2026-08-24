"""
================================================================================
  tests/test_docs_match_canonical_metrics.py
  ────────────────────────────────────────────────────────────────────────────
  This project's "single source of truth" problem, named explicitly by the
  user tonight: numbers get hand-computed once, then hand-copied into
  README.md, paysense_report.tex, and several backend docs, and nothing
  automatically checks they still agree with the actual frozen artifacts
  or with each other. That gap caused real, repeated drift tonight --
  three separate corrections to the same headline numbers, and one MORE
  stale reference found and fixed while writing this very test (README's
  "Generalization Check" section still cited the pre-correction raw-
  XGBoost figure after the main correction pass had already happened).

  This is not a full templating system (README/the report are prose
  documents with numbers woven into sentences, not just tables -- auto-
  generating that safely is a bigger, riskier change than this project
  should make in one more pass tonight). It is the next-cheapest thing
  that actually closes the gap that mattered: it recomputes the canonical
  metrics directly from the on-disk artifacts (the same way
  test_frozen_model_metrics.py already does) and parses the *exact*
  headline numbers out of README.md's Key Results table and
  paysense_report.tex's results table, asserting they match. A future
  retrain or threshold change will make this test fail loudly and
  specifically -- "README says X, artifacts say Y" -- instead of relying
  on someone noticing by chance, which is exactly how tonight's drift
  went undetected for as long as it did each time.
================================================================================
"""
import pathlib
import re

import joblib
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent
ARTEFACTS_DIR = BASE_DIR / "artefacts"
MASTER_CSV = BASE_DIR / "paysense_master_dataset.csv"
README = REPO_ROOT / "README.md"
REPORT_TEX = REPO_ROOT / "PaySense-Report" / "paysense_report.tex"

RANDOM_STATE = 42
DROP_COLS = [
    "transaction_id", "user_id", "receiver_id", "timestamp", "date",
    "data_source", "user_kyc_status", "status", "usr_home_city",
]
TOL_AUC = 0.005   # ROC-AUC / PR-AUC: tight, these should match almost exactly
TOL_PCT = 1.0     # precision/recall reported as a percentage: allow 1 percentage point


def _skip_if_missing(*paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"Required file(s) not present in this environment: {missing}")


@pytest.fixture(scope="module")
def canonical_metrics():
    """The actual source of truth: recomputed directly from the frozen
    artifacts + real ensemble, the same way test_frozen_model_metrics.py
    does. If this and that file's PUBLISHED_* constants ever disagree,
    that file is the one that needs updating -- this fixture is
    intentionally independent so the two can cross-check each other."""
    _skip_if_missing(
        ARTEFACTS_DIR / "paysense_model.pkl",
        ARTEFACTS_DIR / "paysense_preprocessor.pkl",
        ARTEFACTS_DIR / "paysense_threshold.pkl",
        MASTER_CSV,
    )
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from src import fraud_model

    df = pd.read_csv(MASTER_CSV).drop(columns=DROP_COLS)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)
    _, X_test_raw, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    threshold = joblib.load(ARTEFACTS_DIR / "paysense_threshold.pkl")
    fraud_model.load_artefacts()
    records = X_test_raw.to_dict(orient="records")
    y_proba = [fraud_model.score(rec).ensemble_score for rec in records]
    y_pred = [1 if s >= threshold else 0 for s in y_proba]

    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "precision_pct": precision_score(y_test, y_pred, zero_division=0) * 100,
        "recall_pct": recall_score(y_test, y_pred, zero_division=0) * 100,
    }


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


class TestReadmeKeyResultsMatchesCanonical:
    def test_roc_auc(self, canonical_metrics):
        _skip_if_missing(README)
        text = _read(README)
        m = re.search(r"\|\s*ROC-AUC.*?\*\*([\d.]+)\*\*", text)
        assert m, "Could not find a ROC-AUC row in README.md's Key Results table -- table format changed?"
        published = float(m.group(1))
        assert published == pytest.approx(canonical_metrics["roc_auc"], abs=TOL_AUC), (
            f"README.md's Key Results table says ROC-AUC={published}, but the "
            f"on-disk artifacts + real ensemble actually produce "
            f"{canonical_metrics['roc_auc']:.4f}. Recompute and update README.md."
        )

    def test_pr_auc(self, canonical_metrics):
        _skip_if_missing(README)
        text = _read(README)
        m = re.search(r"\|\s*PR-AUC.*?\*\*([\d.]+)\*\*", text)
        assert m, "Could not find a PR-AUC row in README.md's Key Results table -- table format changed?"
        published = float(m.group(1))
        assert published == pytest.approx(canonical_metrics["pr_auc"], abs=TOL_AUC), (
            f"README.md's Key Results table says PR-AUC={published}, but the "
            f"on-disk artifacts + real ensemble actually produce "
            f"{canonical_metrics['pr_auc']:.4f}. Recompute and update README.md."
        )

    def test_deployed_threshold(self, canonical_metrics):
        _skip_if_missing(README)
        text = _read(README)
        m = re.search(r"deployed threshold \(t=([\d.]+)\)", text)
        assert m, "Could not find 'deployed threshold (t=X)' in README.md -- wording changed?"
        published = float(m.group(1))
        assert published == pytest.approx(canonical_metrics["threshold"], abs=1e-6), (
            f"README.md says the deployed threshold is {published}, but "
            f"artefacts/paysense_threshold.pkl is actually "
            f"{canonical_metrics['threshold']}. Recompute and update README.md."
        )

    def test_precision_at_threshold(self, canonical_metrics):
        _skip_if_missing(README)
        text = _read(README)
        m = re.search(r"Precision @ deployed threshold[^|]*\|\s*\*\*([\d.]+)%\*\*", text)
        assert m, "Could not find the Precision row in README.md's Key Results table -- format changed?"
        published = float(m.group(1))
        assert published == pytest.approx(canonical_metrics["precision_pct"], abs=TOL_PCT), (
            f"README.md says precision={published}%, but recomputing against "
            f"the real ensemble gives {canonical_metrics['precision_pct']:.2f}%. "
            f"Recompute and update README.md."
        )

    def test_recall_at_threshold(self, canonical_metrics):
        _skip_if_missing(README)
        text = _read(README)
        m = re.search(r"Recall @ deployed threshold[^|]*\|\s*\*\*([\d.]+)%\*\*", text)
        assert m, "Could not find the Recall row in README.md's Key Results table -- format changed?"
        published = float(m.group(1))
        assert published == pytest.approx(canonical_metrics["recall_pct"], abs=TOL_PCT), (
            f"README.md says recall={published}%, but recomputing against "
            f"the real ensemble gives {canonical_metrics['recall_pct']:.2f}%. "
            f"Recompute and update README.md."
        )


class TestIeeeReportResultsTableMatchesCanonical:
    def test_roc_auc(self, canonical_metrics):
        _skip_if_missing(REPORT_TEX)
        text = _read(REPORT_TEX)
        m = re.search(r"ROC-AUC\s*&\s*([\d.]+)\s*\\\\", text)
        assert m, "Could not find 'ROC-AUC & X \\\\' in paysense_report.tex's results table -- format changed?"
        published = float(m.group(1))
        assert published == pytest.approx(canonical_metrics["roc_auc"], abs=TOL_AUC), (
            f"paysense_report.tex's results table says ROC-AUC={published}, "
            f"but the on-disk artifacts + real ensemble actually produce "
            f"{canonical_metrics['roc_auc']:.4f}. Recompute and update the report."
        )

    def test_pr_auc(self, canonical_metrics):
        _skip_if_missing(REPORT_TEX)
        text = _read(REPORT_TEX)
        m = re.search(r"PR-AUC \(primary\)\s*&\s*([\d.]+)\s*\\\\", text)
        assert m, "Could not find 'PR-AUC (primary) & X \\\\' in paysense_report.tex -- format changed?"
        published = float(m.group(1))
        assert published == pytest.approx(canonical_metrics["pr_auc"], abs=TOL_AUC), (
            f"paysense_report.tex's results table says PR-AUC={published}, "
            f"but the on-disk artifacts + real ensemble actually produce "
            f"{canonical_metrics['pr_auc']:.4f}. Recompute and update the report."
        )

    def test_deployed_threshold(self, canonical_metrics):
        _skip_if_missing(REPORT_TEX)
        text = _read(REPORT_TEX)
        m = re.search(r"Deployed Threshold \$\\tau\^\* = ([\d.]+)\$", text)
        assert m, "Could not find the results table caption's tau^* value in paysense_report.tex -- format changed?"
        published = float(m.group(1))
        assert published == pytest.approx(canonical_metrics["threshold"], abs=1e-6), (
            f"paysense_report.tex says the deployed threshold is {published}, "
            f"but artefacts/paysense_threshold.pkl is actually "
            f"{canonical_metrics['threshold']}. Recompute and update the report."
        )
