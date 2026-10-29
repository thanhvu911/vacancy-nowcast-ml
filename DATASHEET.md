# Datasheet: County Vacancy Panel (Census ACS vs. USPS), 2010–2024

Format follows *Datasheets for Datasets* (Gebru et al.), as used in my Machine Learning course.

## Motivation
- **Purpose.** To test whether USPS vacancy data can track, and then help predict, the Census
  Bureau's annual county vacancy rate, which arrives about nine months after the year ends.
- **Created by.** Thanh Vu, as an independent project connected to my work as a Data Analyst
  Fellow at EDC Lancaster County's Center for Regional Analysis.
- **Funding.** None.

## Composition
- **Instances.** One row per county-year: 6 counties × 14 years = 84 rows (78 in the model
  table, because each county's first year has no previous year).
- **Counties.** Lancaster County, PA (the county EDC serves) plus five comparison counties:
  Champaign (IL), Cook (IL), King (WA), Montgomery (OH), Onondaga (NY). The comparison counties
  were drawn at random from U.S. counties outside Pennsylvania with population ≥ 65,000, the
  threshold for ACS 1-year estimates.
- **Fields.** Total, occupied and vacant housing units (ACS B25002); Census vacancy rate; USPS
  narrow and broad vacancy rates; engineered lags, changes and flags (see `src/pipeline.py`).
- **Missing data.** No ACS 1-year release for 2020, so that year is absent for every county.
  ACS margins of error were not downloaded.
- **Is it a sample?** Yes. Six counties is not representative of the United States. Larger
  urban counties are over-represented (Cook, King).
- **Confidential or personal data.** None. All values are public, county-level aggregates.

## Collection process
- **Census.** Downloaded from the Census Bureau: ACS 1-year estimates, table B25002, one CSV per
  county-year.
- **USPS.** HUD Aggregated USPS Administrative Data on Address Vacancies
  (huduser.gov/portal/datasets/usps.html), quarterly files 2010–2024, aggregated to county and
  averaged over four quarters per year in the companion project's Microsoft Fabric pipeline.
- **Timeframe.** Assembled in September 2026, covering data years 2010–2024.

## Preprocessing, cleaning and labeling
- The Census vacancy rate is recomputed from raw counts and checked against the published
  dashboard values (maximum difference 0.00005).
- USPS file headers changed case around 2014. A Power Query step silently dropped a filename
  column. Both were fixed in the companion pipeline before the annual table was built.
- **2012 USPS definition change.** The broad rate jumps 2.2–4.9 points in every county in 2012
  because USPS reclassified "No-Stat" addresses. The broad-rate change for 2012 is set to zero
  and a post-2012 flag is added.
- Raw files are kept unchanged in `data/raw/`, and each pipeline layer is saved separately.

## Uses
- **Used for.** Comparing USPS and Census vacancy measures. Evaluating nowcasting models under
  random, time-ordered and leave-one-county-out validation.
- **Should not be used for.** Ranking counties, making claims about any single neighborhood, or
  treating USPS rates as official vacancy estimates. With 78 rows, model results are
  small-sample and should be read as a method demonstration.

## Distribution and maintenance
- Public on GitHub under the MIT license. The underlying Census and HUD data are public domain.
- Maintained by Thanh Vu. The pipeline can be extended to more counties by adding raw files.
