import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from engine.extract import parse_name, pdf_to_text
from engine.industry import add_industry_adjusted, list_packs, norm_ind
from engine.textproc import (build_template, classify, env_sentences, filing_metrics, filter_template, load_keywords,
                             score_sentences, split_sentences, strip_headers)

st.set_page_config(page_title="Greenwash-India", layout="wide")
st.title("Environmental disclosure quality — Indian filings")
st.caption("Specificity · novelty · proportionality, adapted from Liu et al. (2026), Journal of Business Ethics")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Novelty rule", ["paper", "similarity"],
                    format_func=lambda m: "Paper: ≥ N words differ" if m == "paper" else "TF-IDF cosine < threshold")
    wd_thr = st.number_input("N (word-difference threshold)", 1, 10, 2)
    cos_thr = st.slider("Cosine threshold", 0.50, 0.99, 0.80, 0.01)
    kw_file = st.file_uploader("Custom keyword CSV (column: keyword)", type="csv")
    use_tmpl = st.checkbox("Remove BRSR/BRR template & question text", value=True)
    cov_thr = st.slider("Template-match threshold", 0.50, 1.00, 0.80, 0.05, disabled=not use_tmpl)
    extra_tmpl = st.file_uploader("Extra template files (blank BRR/BRSR, etc.)", type=["pdf", "txt"],
                                  accept_multiple_files=True)
    st.subheader("India / industry")
    map_file = st.file_uploader("Industry map CSV (firm, industry, high_impact)", type="csv")
    packs = list_packs()
    manual_packs = st.multiselect("Add industry keyword packs for all firms", list(packs))
    chain_br = st.checkbox("Chain BRR → BRSR as one 'BR' series", value=True)
    min_sent = st.number_input("Min. environmental sentences per filing", 0, 100, 10)
params = dict(mode=mode, cos_thr=cos_thr, wd_thr=wd_thr)
keywords = load_keywords(kw_file)
industry_map = {}
if map_file:
    _imap = pd.read_csv(map_file)
    _imap.columns = [c.strip().lower() for c in _imap.columns]
    for _, _r in _imap.iterrows():
        industry_map[str(_r["firm"]).strip().lower()] = (str(_r["industry"]).strip(), _r.get("high_impact", pd.NA))


def firm_info(firm):
    ind, hi = industry_map.get(firm.lower(), ("Unclassified", pd.NA))
    return {"industry": ind, "high_impact": hi}


def keywords_for(firm):
    extra = set(packs.get(norm_ind(firm_info(firm)["industry"]), []))
    for p in manual_packs:
        extra |= set(packs[p])
    return sorted(set(keywords) | extra)


@st.cache_data(show_spinner=False)
def template_set(extra):
    texts = [p.read_text(encoding="utf-8", errors="ignore")
             for p in sorted((Path(__file__).parent / "data" / "templates").glob("*.txt"))]
    for name, data in extra:
        texts.append(data.decode("utf-8", "ignore") if name.lower().endswith(".txt") else pdf_to_text(io.BytesIO(data)))
    return frozenset(build_template(texts))


tmpl = template_set(tuple((u.name, u.getvalue()) for u in extra_tmpl)) if use_tmpl else frozenset()


def collect_files(source):
    if source == "Upload files":
        ups = st.file_uploader("Name files Firm_Year_DocType.pdf, e.g. TCS_2023_BRSR.pdf",
                               type=["pdf", "txt"], accept_multiple_files=True)
        return [(u.name, u.getvalue()) for u in ups]
    folder = st.text_input("Root folder (one subfolder per firm)")
    if folder and Path(folder).is_dir():
        paths = [p for p in Path(folder).rglob("*") if p.suffix.lower() in (".pdf", ".txt")]
        st.write(f"{len(paths)} files found")
        return [(p.name, p) for p in paths]
    return []


def run(files):
    docs = []
    for name, src in files:
        meta = parse_name(name)
        if meta:
            firm, year, dtype = meta
            if chain_br and dtype.upper() in ("BRR", "BRSR"):
                dtype = "BR"
            docs.append((firm, year, dtype, name, src))
        else:
            st.warning(f"Skipped {name}: expected Firm_Year_DocType")
    docs.sort(key=lambda d: (d[0], d[2], d[1]))
    raw, removed, prior, bar = {}, {}, {}, st.progress(0.0)
    for i, (firm, year, dtype, name, src) in enumerate(docs):
        try:
            data = src if isinstance(src, bytes) else src.read_bytes()
            text = data.decode("utf-8", "ignore") if name.lower().endswith(".txt") else pdf_to_text(io.BytesIO(data))
        except Exception as e:
            st.warning(f"Skipped {name}: {e}")
            continue
        if use_tmpl:
            text = strip_headers(text)
        sents = env_sentences(split_sentences(text), keywords_for(firm))
        removed_rows = []
        if use_tmpl:
            sents, removed_rows = filter_template(sents, tmpl, cov_thr)
        removed[(firm, year, dtype)] = pd.DataFrame(removed_rows, columns=["sentence", "reason"])
        raw[(firm, year, dtype)] = score_sentences(sents, prior.get((firm, dtype)))  # prior = previous year, same doc type
        prior[(firm, dtype)] = sents
        bar.progress((i + 1) / len(docs))
    return raw, removed


def firm_tables(raw, firm):
    keys = [k for k in raw if k[0] == firm]
    sent = pd.concat([classify(raw[k], **params).assign(firm=k[0], year=k[1], doctype=k[2]) for k in keys],
                     ignore_index=True)
    met = pd.DataFrame([{"firm": k[0], "year": k[1], "doctype": k[2], **firm_info(k[0]),
                         **filing_metrics(classify(raw[k], **params))} for k in keys])
    met["low_text"] = (met["quantity"] < min_sent).astype(bool)
    return sent, met


def firm_removed(firm):
    parts = [df.assign(firm=k[0], year=k[1], doctype=k[2])
             for k, df in st.session_state.get("removed", {}).items() if k[0] == firm]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["sentence", "reason"])


tab1, tab2, tab3 = st.tabs(["Score documents", "Trends & performance link", "Methodology"])

with tab1:
    source = st.radio("Source", ["Upload files", "Local folder"], horizontal=True)
    files = collect_files(source)
    if st.button("Run", type="primary", disabled=not files):
        st.session_state["raw"], st.session_state["removed"] = run(files)
    raw = st.session_state.get("raw")
    if raw:
        firms = sorted({k[0] for k in raw})
        tabs = {f: firm_tables(raw, f) for f in firms}
        metrics = add_industry_adjusted(pd.concat([m for _, m in tabs.values()], ignore_index=True))
        tabs = {f: (s, metrics[metrics["firm"] == f]) for f, (s, _) in tabs.items()}
        st.session_state["metrics"] = metrics
        st.dataframe(metrics)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for f, (s, m) in tabs.items():
                z.writestr(f"{f}/{f}_sentences.csv", s.to_csv(index=False))
                z.writestr(f"{f}/{f}_filing_metrics.csv", m.to_csv(index=False))
                z.writestr(f"{f}/{f}_removed_template.csv", firm_removed(f).to_csv(index=False))
        st.download_button("Download per-firm files (zip)", buf.getvalue(), "greenwash_india_output.zip")
        if source == "Local folder":
            out_dir = st.text_input("Output folder for per-firm CSVs")
            if st.button("Write to folder", disabled=not out_dir):
                for f, (s, m) in tabs.items():
                    d = Path(out_dir) / f
                    d.mkdir(parents=True, exist_ok=True)
                    s.to_csv(d / f"{f}_sentences.csv", index=False)
                    m.to_csv(d / f"{f}_filing_metrics.csv", index=False)
                    firm_removed(f).to_csv(d / f"{f}_removed_template.csv", index=False)
                st.success(f"Written to {out_dir}")
        pick = st.selectbox("Inspect sentences for", firms)
        st.dataframe(tabs[pick][0])
        with st.expander("Removed as template/question text (audit)"):
            st.dataframe(firm_removed(pick))

with tab2:
    m = st.session_state.get("metrics")
    if m is None:
        st.info("Run scoring first.")
    else:
        base = ["specificity_ratio", "novelty_ratio", "fluff_ratio"]
        sfx = "" if st.radio("Measures", ["Raw", "Industry-adjusted (z-score)"], horizontal=True) == "Raw" else "_ind_z"
        st.line_chart(m.groupby("year")[[b + sfx for b in base]].mean())
        st.caption("Cross-firm means by year. Novelty/fluff need a prior filing of the same firm and document type. "
                   "Adjusted measures need >= 3 classified peers per industry-year.")
        st.write("Industry means (raw)")
        st.bar_chart(m.groupby("industry")[base].mean())
        perf = st.file_uploader("Performance CSV (columns: firm, year, plus numeric variables such as "
                                "emissions intensity or controversy count)", type="csv")
        if perf:
            p = pd.read_csv(perf)
            col = st.selectbox("Performance variable", [c for c in p.columns if c not in ("firm", "year")])
            y = st.selectbox("Text measure", [b + sfx for b in base] + ([] if sfx else ["signal_to_noise"]))
            j = m.merge(p, on=["firm", "year"]).dropna(subset=[col, y])
            st.scatter_chart(j, x=col, y=y)
            if len(j) > 2:
                st.write(f"Pearson r = {j[col].corr(j[y]):.3f}  (n = {len(j)})")
            st.caption("Descriptive only: a talk–walk gap needs controls, firm/industry-year effects and clustered SEs.")

with tab3:
    st.markdown("""
**Unit:** environmental sentences (keyword match, editable in `data/env_keywords.csv`).

**Template removal:** questions (ending in '?'), form instructions ('provide details in the following format', 'If yes...'), 'Principle N: Businesses should...' headings, and text copied from the blank BRSR/BRR formats (`data/templates/`, plus any uploaded template) are dropped before scoring and listed in the audit table.

**Specificity:** a sentence is *specific* if it has a number (years and 'Principle 6'-type references ignored), ₹/crore/lakh,
a unit (tCO₂e, kL, MW, %...), an Indian regulator/standard (CPCB, NGT, ISO 14001...), a state/city/site, or a named entity.

**Novelty:** each sentence is compared with the firm's nearest sentence in its previous filing of the same document type.
*Paper rule:* novel if ≥ N words differ. *Similarity rule:* novel if TF-IDF cosine < threshold.

**Proportionality:** *useful* = specific or novel; *fluff* = general and repetitive;
`signal_to_noise = useful / fluff` (a monotone transform of `fluff_ratio`, so report one of them).

**India & industries:** each firm's industry comes from the industry map CSV; an industry can add its own keywords (`data/industries/<industry>.csv`; add a file to add an industry). `_ind_dm`, `_ind_z`, `_ind_pct` compare a filing with industry peers in the same year (>= 3 peers; filings under the minimum sentence count are excluded). Use these, not raw ratios, across industries. Years follow your filenames (e.g. 2023 = FY2022-23); BRR and BRSR are chained as one 'BR' series.

**Not measured:** truthfulness, sincerity or actual environmental performance. Greenwashing requires linking these
text measures to performance (Tab 2). Scanned PDFs need OCR; BRSR numeric tables are better taken from structured/XBRL data.
""")
