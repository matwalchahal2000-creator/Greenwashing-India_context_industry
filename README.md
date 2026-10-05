# Greenwash-India

Streamlit app scoring environmental disclosure quality (specificity, novelty, proportionality) in Indian
company filings (BRSR, BRR, annual report/MD&A), adapted from Liu et al. (2026, *J. Business Ethics*).

## Input convention
`Firm_Year_DocType.pdf` (or `.txt`), one subfolder per firm, e.g. `TCS/TCS_2023_BRSR.pdf`.
Novelty compares each filing with the firm's previous year of the same DocType; the first year has no novelty.

## Run locally (direct folder access)
    pip install -r requirements.txt
    streamlit run app.py

## Deploy from GitHub
1. `git init && git add . && git commit -m "init"`, create a GitHub repo, `git push`.
2. Go to share.streamlit.io, sign in with GitHub, **Create app**, pick the repo/branch, main file `app.py`, Deploy.
3. On the cloud, use *Upload files* (size/memory limits apply; run large annual reports locally).

## Output
Per firm: `<Firm>/<Firm>_sentences.csv` and `<Firm>/<Firm>_filing_metrics.csv` (zip download, or written to a local folder).
Concatenate the metrics files for panel regressions.

## Extend
Edit `data/env_keywords.csv`; add states/regulators in `engine/textproc.py`; merge performance data in Tab 2.

## Template removal
BRSR/BRR question text, instructions and principle headings are removed before scoring (`data/templates/*.txt` hold the blank formats; upload more in the sidebar). Check the *Removed* audit table and tune the threshold.

## India & industries
- Optional `industry_map.csv` (columns `firm, industry, high_impact`; see `data/industry_map_example.csv`). Firm names must match the filenames; `high_impact` (1/0) can follow CPCB's categorisation.
- Industry keyword packs live in `data/industries/<industry>.csv`. To add an industry, add a CSV with a `keyword` column and use the same name in the map.
- Output adds `industry`, `high_impact`, `low_text` and industry-adjusted columns (`*_ind_dm`, `*_ind_z`, `*_ind_pct`). Use them for cross-industry comparison.
- Name BRR and BRSR files with DocType `BRR`/`BRSR`; they are chained as one `BR` series so novelty carries across the FY2022-23 format change.
