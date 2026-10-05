import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path(__file__).resolve().parent.parent / "data"

# ---------- sentences & environmental filter ----------
def load_keywords(path=None):
    df = pd.read_csv(path or DATA / "env_keywords.csv")
    return [k.strip().lower() for k in df["keyword"].dropna()]


def clean(text):
    return re.sub(r"\s+", " ", re.sub(r"-\n(?=[a-z])", "", text))


_SPLIT = re.compile(r"(?<=[.;!?])\s+(?=[A-Z0-9(\"“‘])|(?<=format:)\s+")


def split_sentences(text, min_words=5, max_words=150):
    return [s.strip() for s in _SPLIT.split(clean(text)) if min_words <= len(s.split()) <= max_words]


def env_sentences(sentences, keywords):
    pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")", re.I)
    return [s for s in sentences if pat.search(s)]


# ---------- specificity (rule-based, India-adapted) ----------
STATES = ("Andhra Pradesh|Assam|Bihar|Chhattisgarh|Delhi|Goa|Gujarat|Haryana|Himachal Pradesh|Jharkhand|"
          "Karnataka|Kerala|Madhya Pradesh|Maharashtra|Odisha|Punjab|Rajasthan|Tamil Nadu|Telangana|"
          "Uttar Pradesh|Uttarakhand|West Bengal|Mumbai|Chennai|Pune|Hyderabad|Bengaluru|Kolkata|"
          "Jamnagar|Mundra|Dahej|Surat|Vadodara|Nagpur|Kochi|Visakhapatnam|Raipur|Jamshedpur|Korba|Singrauli|Angul|"
          "Manipur|Meghalaya|Tripura|Sikkim|Jammu|Ladakh|Puducherry|Chandigarh")
_YEAR = re.compile(r"\b(?:19|20)\d{2}(?:-\d{2,4})?\b")
_REF = re.compile(r"\b(?:Principle|Section|Question|Annexure|Note|Table|Figure|Para)\s*\d+\b", re.I)
_PATTERNS = {
    "has_number": re.compile(r"\d"),
    "has_currency": re.compile(r"₹|\bRs\.?\s?\d|\bINR\b|\bUSD\b|\$|\bcrore\b|\blakh\b", re.I),
    "has_unit": re.compile(r"%|\b(?:tCO2e?|MtCO2e|kL|MLD|MW|GW|MWh|GWh|kWh|GJ|TJ|tonnes?|MT|kg)\b"),
    "has_regulator": re.compile(r"\b(?:CPCB|SPCB|MoEFCC|NGT|BEE|ISO ?14001|ISO ?50001|GRI|SBTi|TCFD|CDP|EIA|ZLD|ETP|CAQM|CCTS|CRZ)\b|(?i:pollution control board|consent to (?:operate|establish)|(?:environmental|forest) clearance|green credit|pat scheme)"),
    "has_location": re.compile(rf"\b(?:{STATES})\b"),
    "has_entity": re.compile(r"(?<=[a-z,;] )[A-Z][A-Za-z&.]+(?: (?:of |and )?[A-Z][A-Za-z&.]+)+"),
}


def specificity_flags(s):
    no_years = _REF.sub(" ", _YEAR.sub(" ", s))  # years / 'Principle 6' are not data
    f = {k: bool(p.search(no_years if k == "has_number" else s)) for k, p in _PATTERNS.items()}
    f["specific"] = any(f.values())
    return f


# ---------- novelty vs. the firm's previous filing ----------
def novelty_vs_prior(curr, prior):
    """Per sentence: cosine to nearest prior sentence and number of differing words."""
    if not prior or not curr:
        return [(np.nan, np.nan)] * len(curr)
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(curr + prior)
    sim = (vec.transform(curr) @ vec.transform(prior).T).toarray()
    out = []
    for i, s in enumerate(curr):
        j = int(sim[i].argmax())
        a, b = s.lower().split(), prior[j].lower().split()
        matched = sum(m.size for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
        out.append((float(sim[i, j]), max(len(a), len(b)) - matched))
    return out


def score_sentences(sents, prior_sents=None):
    nov = novelty_vs_prior(sents, prior_sents)
    rows = [{"sentence": s, **specificity_flags(s), "nn_cosine": c, "word_diff": w}
            for s, (c, w) in zip(sents, nov)]
    return pd.DataFrame(rows)


def classify(df, mode="paper", cos_thr=0.80, wd_thr=2):
    """Apply novelty rule -> novel / useful (specific or novel) / fluff (general and repetitive)."""
    if df.empty:
        return df
    df = df.copy()
    if mode == "paper":
        df["novel"] = np.where(df["word_diff"].isna(), np.nan, (df["word_diff"] >= wd_thr).astype(float))
    else:
        df["novel"] = np.where(df["nn_cosine"].isna(), np.nan, (df["nn_cosine"] < cos_thr).astype(float))
    df["useful"] = df["specific"] | (df["novel"] == 1)
    df["fluff"] = (~df["specific"]) & (df["novel"] == 0)
    return df


def filing_metrics(df):
    nan = float("nan")
    n = len(df)
    m = dict(quantity=n, specific=0, general=0, specificity_ratio=nan, novel=nan, repetitive=nan,
             novelty_ratio=nan, useful=nan, fluff=nan, fluff_ratio=nan, signal_to_noise=nan)
    if n == 0:
        return m
    m["specific"] = int(df["specific"].sum())
    m["general"] = n - m["specific"]
    m["specificity_ratio"] = m["specific"] / n
    if df["novel"].notna().any():  # needs a prior filing (first filing per firm/doc type has none)
        nv, us, fl = int((df["novel"] == 1).sum()), int(df["useful"].sum()), int(df["fluff"].sum())
        m.update(novel=nv, repetitive=n - nv, novelty_ratio=nv / n, useful=us, fluff=fl,
                 fluff_ratio=fl / n, signal_to_noise=us / fl if fl else nan)
    return m


# ---------- BRSR / BRR template & question removal ----------
_PRINCIPLES = [
    "Businesses should conduct and govern themselves with integrity, and in a manner that is Ethical, Transparent and Accountable",
    "Businesses should provide goods and services in a manner that is sustainable and safe",
    "Businesses should respect and promote the well-being of all employees, including those in their value chains",
    "Businesses should respect the interests of and be responsive to all its stakeholders",
    "Businesses should respect and promote human rights",
    "Businesses should respect and make efforts to protect and restore the environment",
    "Businesses, when engaging in influencing public and regulatory policy, should do so in a manner that is responsible and transparent",
    "Businesses should promote inclusive growth and equitable development",
    "Businesses should engage with and provide value to their consumers in a responsible manner",
    # older BRR (NVG 2011) wording
    "Businesses should conduct and govern themselves with Ethics, Transparency and Accountability",
    "Businesses should provide goods and services that are safe and contribute to sustainability throughout their life cycle",
    "Businesses should promote the wellbeing of all employees",
    "Businesses should respect the interests of, and be responsive towards all stakeholders, especially those who are disadvantaged, vulnerable and marginalised",
    "Businesses should respect, protect, and make efforts to restore the environment",
    "Businesses, when engaged in influencing public and regulatory policy, should do so in a responsible manner",
    "Businesses should support inclusive growth and equitable development",
    "Businesses should engage with and provide value to their customers and consumers in a responsible manner",
]


def _words_re(p):
    return r"[\s,]+".join(map(re.escape, re.findall(r"[A-Za-z0-9\-]+", p)))


_PRINC_RE = re.compile(r"(?:principle\s*\d+\s*[:\-–.]?\s*)?(?:" + "|".join(_words_re(p) for p in _PRINCIPLES) + r")\s*[.:]?", re.I)


def strip_headers(text):
    """Remove 'Principle N: Businesses should ...' headings (BRSR and BRR wording)."""
    return _PRINC_RE.sub(" ", text)


_NORM = re.compile(r"[^a-z0-9]+")
_LEAD = re.compile(r"^(?:y\s*/\s*n|yes\s*/\s*no)\s+", re.I)
_INSTR = re.compile(
    r"in the following format|provide (?:the )?following|please (?:specify|provide|give|mention|indicate)|"
    r"indicate whether|\bif (?:yes|no|not)\b[ ,]+(?:then |please )?(?:provide|give|specify|mention|describe|what|whether)|"
    r"\((?:yes ?/ ?no|y ?/ ?n)\b", re.I)


def _shingles(text, n=4):
    t = _NORM.sub(" ", text.lower()).split()
    return {" ".join(t[i:i + n]) for i in range(len(t) - n + 1)}


def build_template(texts, n=4):
    """Set of word 4-grams found in blank BRSR/BRR format texts."""
    out = set()
    for t in texts:
        out |= _shingles(t, n)
    return out


def template_coverage(s, tmpl, n=4):
    sh = _shingles(s, n)
    return len(sh & tmpl) / len(sh) if sh else 0.0


def filter_template(sentences, tmpl, cov_thr=0.80):
    """Drop questions, form instructions and sentences copied from the blank format.
    Returns (kept, removed) where removed = [(sentence, reason)]."""
    kept, removed = [], []
    for s in sentences:
        s2 = _LEAD.sub("", s).strip()  # 'Y/N Yes, the Company ...' -> the answer
        if s2.endswith("?"):
            reason = "question"
        elif _INSTR.search(s2):
            reason = "instruction"
        elif tmpl and template_coverage(s2, tmpl) >= cov_thr:
            reason = "template-match"
        else:
            reason = None
        if reason:
            removed.append((s, reason))
        else:
            kept.append(s2)
    return kept, removed
