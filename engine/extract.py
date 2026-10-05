import re
from pypdf import PdfReader


def pdf_to_text(file) -> str:
    """Extract text from a text-based PDF (scanned PDFs need OCR first)."""
    pages = []
    reader = PdfReader(file)
    if reader.is_encrypted:
        reader.decrypt("")  # many reports are encrypted with an empty user password
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def parse_name(filename: str):
    """'TCS_2023_BRSR.pdf' -> ('TCS', 2023, 'BRSR'); None if the pattern does not match."""
    stem = re.sub(r"\.(pdf|txt)$", "", filename.replace("\\", "/").split("/")[-1], flags=re.I)
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit():
        return parts[0], int(parts[1]), parts[2]
    return None
