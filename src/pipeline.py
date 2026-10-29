"""
Data pipeline: raw files -> bronze -> silver -> gold, with data-quality checks.

  bronze  every raw Census CSV row, unchanged, plus the file it came from
  silver  typed, one row per county-year, Census vacancy rate recomputed from counts,
          joined to the USPS annual rates
  gold    the modeling table: last year's values, year-over-year changes, and flags,
          built with SQL window functions (SQLite, standard library only)

Every layer is written to data/<layer>/ as CSV so each step can be inspected.
The run stops with an error if any data-quality check fails.

Run:  python src/pipeline.py
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = {layer: ROOT / "data" / layer for layer in ("bronze", "silver", "gold")}

EXPECTED_COUNTIES = {"Champaign", "Cook", "King", "Lancaster", "Montgomery", "Onondaga"}
EXPECTED_YEARS = set(range(2010, 2025)) - {2020}  # no ACS 1-year release for 2020


# ---------------------------------------------------------------------------
# Bronze
# ---------------------------------------------------------------------------
def build_bronze() -> pd.DataFrame:
    frames = []
    for path in sorted((RAW / "census").glob("*/*.csv")):
        m = re.fullmatch(r"(?P<county>[A-Za-z]+)_(?P<fips>\d{5})", path.parent.name)
        df = pd.read_csv(path, dtype=str)
        df["source_file"] = f"{path.parent.name}/{path.name}"
        df["folder_county"], df["folder_fips"] = m["county"], m["fips"]
        df["file_year"] = int(path.stem)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Silver
# ---------------------------------------------------------------------------
def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame({
        "county": bronze["folder_county"],
        "fips": bronze["GEO_ID"].str[-5:],
        "folder_fips": bronze["folder_fips"],
        "year": bronze["file_year"],
        "housing_units": pd.to_numeric(bronze["B25002_001E"]),
        "occupied": pd.to_numeric(bronze["B25002_002E"]),
        "vacant": pd.to_numeric(bronze["B25002_003E"]),
    })
    s["census_rate"] = s["vacant"] / s["housing_units"]
    usps = pd.read_csv(RAW / "usps_county_year.csv")
    s = s.merge(usps[["county", "year", "usps_narrow_rate", "usps_broad_rate"]],
                on=["county", "year"], how="left", validate="one_to_one")
    return s.sort_values(["county", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gold (SQL)
# ---------------------------------------------------------------------------
GOLD_SQL = """
WITH ordered AS (
    SELECT
        county, year, census_rate, usps_narrow_rate, usps_broad_rate,
        LAG(year)             OVER w AS prev_year,
        LAG(census_rate)      OVER w AS census_lag,
        LAG(usps_narrow_rate) OVER w AS usps_narrow_lag,
        LAG(usps_broad_rate)  OVER w AS usps_broad_lag
    FROM silver
    WINDOW w AS (PARTITION BY county ORDER BY year)
)
SELECT
    county,
    year,
    census_rate                       AS census,
    census_lag,
    census_rate - census_lag          AS d_census,
    usps_narrow_rate                  AS usps_narrow,
    usps_broad_rate                   AS usps_broad,
    usps_narrow_rate - usps_narrow_lag AS d_usps_narrow,
    -- USPS redefined "No-Stat" addresses in 2012, so the broad rate jumps 3-5 points in every
    -- county that year. That jump is a definition change, not a vacancy change: zero it out.
    CASE WHEN year = 2012 THEN 0.0 ELSE usps_broad_rate - usps_broad_lag END AS d_usps_broad,
    year - prev_year                  AS gap_years,        -- 2 for 2021 (no 2020 data)
    CASE WHEN year >= 2012 THEN 1 ELSE 0 END AS post_2012,
    CASE WHEN census_rate > census_lag THEN 1 ELSE 0 END AS census_up
FROM ordered
WHERE census_lag IS NOT NULL
ORDER BY county, year
"""


def build_gold(silver: pd.DataFrame) -> pd.DataFrame:
    con = sqlite3.connect(":memory:")
    silver.to_sql("silver", con, index=False)
    gold = pd.read_sql_query(GOLD_SQL, con)
    con.close()
    return gold


# ---------------------------------------------------------------------------
# Data-quality checks. Each returns (name, passed, detail).
# ---------------------------------------------------------------------------
def quality_checks(bronze, silver, gold):
    checks = []

    def add(name, passed, detail=""):
        checks.append((name, bool(passed), detail))

    add("bronze: one row per raw file", len(bronze) == bronze["source_file"].nunique(),
        f"{len(bronze)} rows, {bronze['source_file'].nunique()} files")
    add("bronze: no 'N' / '-' placeholders in count columns",
        not bronze[["B25002_001E", "B25002_002E", "B25002_003E"]].isin(["N", "-", "(X)"]).any().any())
    add("silver: GEO_ID FIPS matches folder FIPS", (silver["fips"] == silver["folder_fips"]).all())
    add("silver: expected six counties", set(silver["county"]) == EXPECTED_COUNTIES,
        ", ".join(sorted(silver["county"].unique())))
    years_ok = silver.groupby("county")["year"].apply(lambda y: set(y) == EXPECTED_YEARS).all()
    add("silver: 14 years per county (2010-2024, no 2020)", years_ok)
    add("silver: no duplicate county-years", not silver.duplicated(["county", "year"]).any())
    add("silver: occupied + vacant = total housing units",
        (silver["occupied"] + silver["vacant"] == silver["housing_units"]).all())
    add("silver: rates between 0 and 1",
        silver[["census_rate", "usps_narrow_rate", "usps_broad_rate"]].apply(lambda c: c.between(0, 1)).all().all())
    add("silver: every county-year has USPS data", silver[["usps_narrow_rate", "usps_broad_rate"]].notna().all().all())
    add("silver: USPS narrow rate below broad rate (broad adds No-Stat)",
        (silver["usps_narrow_rate"] < silver["usps_broad_rate"]).all())

    ref = pd.read_csv(RAW / "reference_census_rate_from_dashboard.csv")
    m = silver.merge(ref, on=["county", "year"])
    max_diff = (m["census_rate"] - m["census_rate_published"]).abs().max()
    add("silver: recomputed Census rate matches the published dashboard (within 0.0005)",
        len(m) == len(silver) and max_diff <= 0.0005, f"max difference {max_diff:.5f}")

    add("gold: 78 modeling rows (84 minus each county's first year)", len(gold) == 78, f"{len(gold)} rows")
    add("gold: no missing values", not gold.isna().any().any())
    add("gold: only 2021 has a two-year gap", set(gold.loc[gold["gap_years"] == 2, "year"]) == {2021})
    jump = (silver[silver["year"] == 2012].set_index("county")["usps_broad_rate"]
            - silver[silver["year"] == 2011].set_index("county")["usps_broad_rate"])
    add("note: 2012 USPS broad-rate jump present in every county (definition change)",
        (jump > 0.02).all(), "jumps of " + ", ".join(f"{v*100:.1f}pp" for v in jump))
    return checks


def main():
    for p in OUT.values():
        p.mkdir(parents=True, exist_ok=True)
    bronze = build_bronze()
    silver = build_silver(bronze)
    gold = build_gold(silver)
    bronze.to_csv(OUT["bronze"] / "census_raw_union.csv", index=False)
    silver.to_csv(OUT["silver"] / "county_year.csv", index=False)
    gold.to_csv(OUT["gold"] / "model_table.csv", index=False)

    checks = quality_checks(bronze, silver, gold)
    lines = ["| Check | Result | Detail |", "|---|---|---|"]
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_quality_report.md").write_text("\n".join(lines) + "\n")
    failed = [n for n, ok, _ in checks if not ok]
    if failed:
        raise SystemExit(f"{len(failed)} data-quality check(s) failed: {failed}")
    print(f"\nbronze {len(bronze)} rows -> silver {len(silver)} rows -> gold {len(gold)} rows")


if __name__ == "__main__":
    main()
