import re
from pathlib import Path

import pandas as pd

IND = Path(__file__).resolve().parent.parent / "data" / "industries"
MEASURES = ["specificity_ratio", "novelty_ratio", "fluff_ratio"]


def norm_ind(x):
    return re.sub(r"\W+", "_", str(x).lower()).strip("_")


def list_packs():
    """One CSV per industry in data/industries/ (column: keyword). Add a file to add an industry."""
    return {p.stem: [k.strip().lower() for k in pd.read_csv(p)["keyword"].dropna()] for p in sorted(IND.glob("*.csv"))}


def add_industry_adjusted(m, min_peers=3):
    """Within industry-year demeaned (_ind_dm), z-score (_ind_z) and percentile (_ind_pct) versions of the
    text measures. Low-text filings and unclassified firms are excluded from peer groups."""
    m = m.copy()
    ind = m["industry"].where(m["industry"] != "Unclassified")
    for c in MEASURES:
        x = m[c].where(~m["low_text"].astype(bool))
        g = x.groupby([ind, m["year"]])
        n, mu, sd = g.transform("count"), g.transform("mean"), g.transform("std")
        m[f"{c}_ind_dm"] = (x - mu).where(n >= 2)
        m[f"{c}_ind_z"] = ((x - mu) / sd).where(n >= min_peers)
        m[f"{c}_ind_pct"] = g.rank(pct=True).where(n >= min_peers)
    return m
