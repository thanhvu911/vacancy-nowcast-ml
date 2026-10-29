import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gd import LinearRegressionGD, LogisticRegressionGD  # noqa: E402
import pipeline  # noqa: E402


def test_linear_gd_matches_closed_form():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)) * [1, 10, 0.01]  # very different scales on purpose
    y = 2 + X @ np.array([0.5, -0.03, 40.0]) + rng.normal(scale=0.1, size=200)
    gd = LinearRegressionGD(learning_rate=0.1, n_iter=20000).fit(X, y)
    A = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    w, b = gd.coef_original_scale_
    assert b == pytest.approx(beta[0], abs=1e-6)
    assert w == pytest.approx(beta[1:], rel=1e-5)


def test_linear_gd_diverges_with_huge_learning_rate():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    y = X @ [1.0, 2.0]
    gd = LinearRegressionGD(learning_rate=5.0, n_iter=200).fit(X, y)
    assert gd.cost_history_[-1] > gd.cost_history_[0] or not np.isfinite(gd.cost_history_[-1])


def test_logistic_gd_matches_sklearn_with_same_penalty():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 2))
    y = (X @ [1.5, -1.0] + rng.normal(size=300) > 0).astype(int)
    l2 = 1.0
    ours = LogisticRegressionGD(learning_rate=0.5, n_iter=20000, l2=l2, tol=1e-14).fit(X, y)
    # Our cost * n = sum cross-entropy + (l2/2)||w||^2, which is sklearn's objective with C = 1/l2.
    ref = make_pipeline(StandardScaler(), LogisticRegression(C=1 / l2, tol=1e-10, max_iter=10000)).fit(X, y)
    assert ours.predict_proba(X) == pytest.approx(ref.predict_proba(X)[:, 1], abs=1e-4)


def test_pipeline_quality_checks_all_pass():
    bronze = pipeline.build_bronze()
    silver = pipeline.build_silver(bronze)
    gold = pipeline.build_gold(silver)
    failed = [name for name, ok, _ in pipeline.quality_checks(bronze, silver, gold) if not ok]
    assert failed == []


def test_gold_has_no_lookahead():
    """census_lag must equal the previous available year's Census rate, never the current one."""
    silver = pipeline.build_silver(pipeline.build_bronze())
    gold = pipeline.build_gold(silver)
    lookup = silver.set_index(["county", "year"])["census_rate"]
    prev_year = gold["year"] - gold["gap_years"]
    expected = [lookup[(c, y)] for c, y in zip(gold["county"], prev_year)]
    assert gold["census_lag"].to_numpy() == pytest.approx(np.array(expected))
