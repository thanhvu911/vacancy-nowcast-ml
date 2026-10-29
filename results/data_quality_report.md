| Check | Result | Detail |
|---|---|---|
| bronze: one row per raw file | PASS | 84 rows, 84 files |
| bronze: no 'N' / '-' placeholders in count columns | PASS |  |
| silver: GEO_ID FIPS matches folder FIPS | PASS |  |
| silver: expected six counties | PASS | Champaign, Cook, King, Lancaster, Montgomery, Onondaga |
| silver: 14 years per county (2010-2024, no 2020) | PASS |  |
| silver: no duplicate county-years | PASS |  |
| silver: occupied + vacant = total housing units | PASS |  |
| silver: rates between 0 and 1 | PASS |  |
| silver: every county-year has USPS data | PASS |  |
| silver: USPS narrow rate below broad rate (broad adds No-Stat) | PASS |  |
| silver: recomputed Census rate matches the published dashboard (within 0.0005) | PASS | max difference 0.00005 |
| gold: 78 modeling rows (84 minus each county's first year) | PASS | 78 rows |
| gold: no missing values | PASS |  |
| gold: only 2021 has a two-year gap | PASS |  |
| note: 2012 USPS broad-rate jump present in every county (definition change) | PASS | jumps of 4.2pp, 3.6pp, 4.0pp, 2.2pp, 3.6pp, 4.9pp |
