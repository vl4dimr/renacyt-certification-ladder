# The certification ladder — replication package

[![DOI](https://zenodo.org/badge/1354238028.svg)](https://doi.org/10.5281/zenodo.22304793)

Replication package for *The certification ladder: what researcher-ranking
levels measure, and what they miss* (submitted to Research Evaluation).

## Contents
- `data/renacyt_limpio.csv` — Peru's public researcher registry (RENACYT,
  cut-off 2024-05-30; CONCYTEC open data, ODC-By), cleaned: includes
  certification level, regulation, gender, and qualification dates.
- `data/autores_pe_enlazados.csv` — 191,089 Peru-affiliated OpenAlex author
  profiles (2015–2026) linked to the registry with the validated three-tier
  name protocol (https://doi.org/10.5281/zenodo.22099851); produced by the
  field-coverage pipeline (https://doi.org/10.5281/zenodo.22160752).
- `scripts/01_exploracion.py` — person-level aggregation and the core audit:
  per-level medians, Spearman correlations, rank-based partials,
  adjacent-rung AUCs, overlap statistics, per-field associations.
- `scripts/02_sensibilidad.py` — robustness: active-only, 2021-regulation
  only, recent-entrant cohort (first publication ≥2018), field-percentile
  normalization.
- `scripts/03_figures.py` — the four manuscript figures (600 dpi) and the
  2021-regulation per-level table.
- `outputs/` — audit results (JSON), per-level tables (CSV), figures.

## Reproduce
```
pip install pandas numpy scipy matplotlib
python scripts/01_exploracion.py
python scripts/02_sensibilidad.py
python scripts/03_figures.py
```

## Licences
Code: MIT. Derived data and outputs: CC BY 4.0. RENACYT source: ODC-By
(CONCYTEC). OpenAlex-derived data: CC0.

## Related
- Linkage-protocol package: https://doi.org/10.5281/zenodo.22099851
- Field-coverage package: https://doi.org/10.5281/zenodo.22160752
- Prior single-field application: https://doi.org/10.5281/zenodo.22070643
