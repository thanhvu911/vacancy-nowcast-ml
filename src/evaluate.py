"""
Evaluate whether USPS data can nowcast the Census vacancy rate, three different ways.

Two tasks, both using only information available before the Census release for year t:
  1. Regression: predict the Census vacancy rate for year t.
  2. Classification: predict whether it went up or down from the previous release.

Three evaluation schemes, applied to every model:
  - Random 5-fold CV (repeated): the number most tutorials report. It lets the model
    train on 2023 to predict 2016, which a real nowcast never could.
  - Forward-chaining: to predict year t, train only on years before t (the real use case).
  - Leave-one-county-out: train on five counties, test on the sixth (a new place).

Run:  python src/pipeline.py && python src/evaluate.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gd import LinearRegressionGD, LogisticRegressionGD

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
SEED, PP = 0, 100  # errors reported in percentage points

ALL_FEATURES = ["census_lag", "usps_narrow", "usps_broad", "d_usps_narrow", "d_usps_broad",
                "gap_years", "post_2012"]
CHANGE_FEATURES = ["d_usps_narrow", "d_usps_broad"]
PERSIST = "Persistence (last Census value)"


def load_gold():
    return pd.read_csv(ROOT / "data" / "gold" / "model_table.csv")


# ---------------------------------------------------------------------------
# Regression models: fit on train, return predicted Census *level* for test.
# "Change" models learn d_census and add it to last year's value.
# ---------------------------------------------------------------------------
def persistence(train, test):
    return test["census_lag"].to_numpy()


def county_average(train, test):
    means = train.groupby("county")["census"].mean()
    return test["county"].map(means).fillna(train["census"].mean()).to_numpy()


def change_model(make, cols):
    def fit_predict(train, test):
        m = make().fit(train[cols].to_numpy(), train["d_census"].to_numpy())
        return test["census_lag"].to_numpy() + m.predict(test[cols].to_numpy())
    return fit_predict


def level_model(make, cols):
    def fit_predict(train, test):
        m = make().fit(train[cols].to_numpy(), train["census"].to_numpy())
        return m.predict(test[cols].to_numpy())
    return fit_predict


REGRESSORS = {
    PERSIST: persistence,
    "County average": county_average,
    "GD linear, USPS change (from scratch)": change_model(lambda: LinearRegressionGD(0.1, 5000), CHANGE_FEATURES),
    "GD ridge, all features (from scratch)": change_model(lambda: LinearRegressionGD(0.1, 5000, l2=10.0), ALL_FEATURES),
    "sklearn OLS, USPS change": change_model(LinearRegression, CHANGE_FEATURES),
    "GD linear, USPS levels only": level_model(lambda: LinearRegressionGD(0.1, 5000),
                                               ["usps_narrow", "usps_broad", "post_2012"]),
    "Random forest, all features": change_model(
        lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED), ALL_FEATURES),
    "Gradient boosting, all features": change_model(
        lambda: HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=150,
                                              min_samples_leaf=5, random_state=SEED), ALL_FEATURES),
}


# ---------------------------------------------------------------------------
# Direction classifiers: return P(Census went up) or a hard 0/1 call.
# ---------------------------------------------------------------------------
def usps_rule(train, test):
    """The original project's test: USPS narrow rate moved the same way as Census."""
    return (test["d_usps_narrow"] > 0).astype(int).to_numpy()


def majority(train, test):
    return np.full(len(test), int(train["census_up"].mean() >= 0.5))


def logistic_gd(cols, l2=1.0):
    def fit_predict(train, test):
        m = LogisticRegressionGD(0.5, 5000, l2=l2).fit(train[cols].to_numpy(), train["census_up"].to_numpy())
        return m.predict(test[cols].to_numpy())
    return fit_predict


def logistic_sklearn(cols):
    def fit_predict(train, test):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        m.fit(train[cols].to_numpy(), train["census_up"].to_numpy())
        return m.predict(test[cols].to_numpy())
    return fit_predict


CLASSIFIERS = {
    "Rule: USPS narrow direction": usps_rule,
    "Majority class": majority,
    "GD logistic, USPS change (from scratch)": logistic_gd(CHANGE_FEATURES),
    "GD logistic, all features (from scratch)": logistic_gd(ALL_FEATURES),
    "sklearn logistic, all features": logistic_sklearn(ALL_FEATURES),
}


# ---------------------------------------------------------------------------
# Evaluation schemes
# ---------------------------------------------------------------------------
def random_kfold(df, n_repeats=20):
    for r in range(n_repeats):
        yield from KFold(5, shuffle=True, random_state=SEED + r).split(df)


def forward_chaining(df, first_test_year=2015):
    for year in sorted(df["year"].unique()):
        if year >= first_test_year:
            yield np.where(df["year"] < year)[0], np.where(df["year"] == year)[0]


def leave_one_county_out(df):
    for c in sorted(df["county"].unique()):
        yield np.where(df["county"] != c)[0], np.where(df["county"] == c)[0]


SCHEMES = {
    "Random 5-fold CV": random_kfold,
    "Forward-chaining (nowcast)": forward_chaining,
    "Leave-one-county-out": leave_one_county_out,
}


def run(df, scheme, models):
    rows = []
    for fold, (tr, te) in enumerate(scheme(df)):
        train, test = df.iloc[tr], df.iloc[te]
        for name, fn in models.items():
            rows.append(pd.DataFrame({
                "fold": fold, "model": name, "county": test["county"].to_numpy(),
                "year": test["year"].to_numpy(), "actual": test["census"].to_numpy(),
                "census_lag": test["census_lag"].to_numpy(), "census_up": test["census_up"].to_numpy(),
                "pred": fn(train, test),
            }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Conformal prediction intervals, calibrated only on earlier years
# ---------------------------------------------------------------------------
def conformal(pred_fc, model, alpha=0.2, min_calib=12):
    p = pred_fc[pred_fc["model"] == model].sort_values("year").copy()
    p["resid"] = (p["actual"] - p["pred"]).abs()
    half, cov = [], []
    for _, r in p.iterrows():
        past = p.loc[p["year"] < r["year"], "resid"].to_numpy()
        if len(past) < min_calib:
            half.append(np.nan); cov.append(np.nan); continue
        n = len(past)
        q = np.quantile(past, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
        half.append(q * PP); cov.append(float(r["resid"] <= q))
    p["halfwidth_pp"], p["covered"] = half, cov
    return p.dropna(subset=["covered"])


# ---------------------------------------------------------------------------
def main():
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    df = load_gold()
    report = {}

    # ---- Regression
    reg_pooled, reg_by_county, reg_preds = [], {}, {}
    for s, scheme in SCHEMES.items():
        p = run(df, scheme, REGRESSORS)
        p["abs_err_pp"] = (p["pred"] - p["actual"]).abs() * PP
        reg_preds[s] = p
        by_c = p.groupby(["model", "county"])["abs_err_pp"].mean().unstack()
        pooled = p.groupby("model")["abs_err_pp"].mean().rename("mae_pp").to_frame()
        pooled["worst_county_mae_pp"] = by_c.max(axis=1)
        pooled["worst_county"] = by_c.idxmax(axis=1)
        pooled["skill_vs_persistence"] = 1 - pooled["mae_pp"] / pooled.loc[PERSIST, "mae_pp"]
        pooled.insert(0, "scheme", s)
        reg_pooled.append(pooled.reset_index())
        reg_by_county[s] = by_c
    reg_pooled = pd.concat(reg_pooled, ignore_index=True)
    reg_pooled.round(4).to_csv(RESULTS / "regression_metrics.csv", index=False)
    for s, by_c in reg_by_county.items():
        by_c.round(3).to_csv(RESULTS / f"regression_mae_by_county__{s.split()[0].lower()}.csv")
    reg_preds["Forward-chaining (nowcast)"].to_csv(RESULTS / "nowcast_predictions.csv", index=False)

    fc = reg_by_county["Forward-chaining (nowcast)"]
    skill_c = 1 - fc.div(fc.loc[PERSIST], axis=1)
    skill_c.round(3).to_csv(RESULTS / "nowcast_skill_by_county.csv")

    # ---- Classification (direction of change)
    cls_rows, conf = [], {}
    for s, scheme in SCHEMES.items():
        p = run(df, scheme, CLASSIFIERS)
        p["correct"] = (p["pred"] == p["census_up"]).astype(int)
        by_c = p.groupby(["model", "county"])["correct"].mean().unstack()
        pooled = p.groupby("model")["correct"].mean().rename("accuracy").to_frame()
        pooled["worst_county_accuracy"] = by_c.min(axis=1)
        pooled["worst_county"] = by_c.idxmin(axis=1)
        pooled["n_predictions"] = p.groupby("model").size()
        pooled.insert(0, "scheme", s)
        cls_rows.append(pooled.reset_index())
        if s == "Forward-chaining (nowcast)":
            by_c.round(3).to_csv(RESULTS / "direction_accuracy_by_county__forward.csv")
            for m in CLASSIFIERS:
                q = p[p["model"] == m]
                conf[m] = confusion_matrix(q["census_up"], q["pred"], labels=[0, 1])
    cls = pd.concat(cls_rows, ignore_index=True)
    cls.round(4).to_csv(RESULTS / "direction_metrics.csv", index=False)
    with open(RESULTS / "confusion_matrices__forward.md", "w") as f:
        f.write("Rows = actual (down, up); columns = predicted (down, up). Forward-chaining test years 2015-2024.\n\n")
        for m, cm in conf.items():
            f.write(f"**{m}**\n\n| | pred down | pred up |\n|---|---|---|\n"
                    f"| actual down | {cm[0,0]} | {cm[0,1]} |\n| actual up | {cm[1,0]} | {cm[1,1]} |\n\n")

    # ---- Intervals
    cand = reg_pooled[(reg_pooled["scheme"] == "Forward-chaining (nowcast)")
                      & ~reg_pooled["model"].isin([PERSIST, "County average"])]
    best = cand.sort_values("mae_pp").iloc[0]["model"]
    cov_tables = []
    for m in [best, PERSIST]:
        c = conformal(reg_preds["Forward-chaining (nowcast)"], m)
        t = c.groupby("county").agg(coverage=("covered", "mean"), halfwidth_pp=("halfwidth_pp", "mean"),
                                    n_years=("covered", "size"))
        t.loc["ALL (pooled)"] = [c["covered"].mean(), c["halfwidth_pp"].mean(), len(c)]
        t.insert(0, "model", m)
        cov_tables.append(t.reset_index())
    coverage = pd.concat(cov_tables, ignore_index=True)
    coverage.round(3).to_csv(RESULTS / "interval_coverage_80pct.csv", index=False)

    # ---- Print a readable summary
    pd.set_option("display.width", 160)
    print("REGRESSION: mean absolute error, percentage points")
    print(reg_pooled.pivot(index="model", columns="scheme", values="mae_pp")[list(SCHEMES)]
          .sort_values("Forward-chaining (nowcast)").round(3).to_string(), "\n")
    print("Nowcast skill vs persistence by county (>0 = beats carrying last year forward)")
    print(skill_c.round(2).to_string(), "\n")
    print("DIRECTION: accuracy")
    print(cls.pivot(index="model", columns="scheme", values="accuracy")[list(SCHEMES)].round(3).to_string(), "\n")
    print(f"80% conformal intervals ({best} vs persistence), forward-chaining")
    print(coverage.round(3).to_string(index=False))

    make_figures(df, reg_pooled, fc, cls, coverage, best)


def make_figures(df, reg_pooled, fc, cls, coverage, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#9aa7b8", "#2f6db3", "#d1793b"]

    pv = reg_pooled.pivot(index="model", columns="scheme", values="mae_pp")[list(SCHEMES)]
    pv = pv.sort_values("Forward-chaining (nowcast)")
    ax = pv.plot.barh(figsize=(8.5, 5), color=colors)
    ax.invert_yaxis(); ax.set_ylabel("")
    ax.set_xlabel("Mean absolute error (percentage points of vacancy rate)")
    ax.set_title("Same models, three evaluation schemes", pad=28)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9)
    plt.tight_layout(); plt.savefig(FIGURES / "1_error_by_scheme.png", dpi=160); plt.close()

    sub = fc.loc[[PERSIST, best]].T.sort_values(best)
    ax = sub.plot.bar(figsize=(8.5, 4), color=colors[:2], rot=0)
    ax.set_ylabel("Nowcast MAE (pp)"); ax.set_xlabel("")
    ax.set_title("Nowcast error by county")
    ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(FIGURES / "2_nowcast_error_by_county.png", dpi=160); plt.close()

    cv = cls.pivot(index="model", columns="scheme", values="accuracy")[list(SCHEMES)]
    ax = cv.sort_values("Forward-chaining (nowcast)").plot.barh(figsize=(8.5, 4), color=colors)
    ax.axvline(0.5, color="black", lw=0.8, ls=":")
    ax.set_xlim(0, 1); ax.set_xlabel("Accuracy calling the direction of change"); ax.set_ylabel("")
    ax.set_title("Did Census vacancy go up or down?", pad=28)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9)
    plt.tight_layout(); plt.savefig(FIGURES / "3_direction_accuracy.png", dpi=160); plt.close()

    c = coverage[coverage["model"] == best].set_index("county")["coverage"]
    ax = c.plot.bar(figsize=(8.5, 3.6), color=[colors[0] if i == "ALL (pooled)" else colors[1] for i in c.index], rot=0)
    ax.axhline(0.8, color=colors[2], ls="--", label="target 80%")
    ax.set_ylim(0, 1.05); ax.set_ylabel("share of years inside interval"); ax.set_xlabel("")
    ax.set_title(f"80% prediction intervals: pooled vs. each county\n({best}, forward-chaining)", fontsize=10)
    ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(FIGURES / "4_interval_coverage_by_county.png", dpi=160); plt.close()

    # Learning-rate behavior of my GD implementation (Project 3 style).
    X = df[CHANGE_FEATURES].to_numpy(); y = df["d_census"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for lr in [0.01, 0.1, 0.5, 1.0]:
        h = LinearRegressionGD(learning_rate=lr, n_iter=300, tol=0).fit(X, y).cost_history_
        ax.plot(h, label=f"learning rate {lr}")
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("cost (MSE / 2)")
    ax.set_title("Gradient descent on the vacancy data"); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(FIGURES / "5_learning_rates.png", dpi=160); plt.close()


if __name__ == "__main__":
    main()
