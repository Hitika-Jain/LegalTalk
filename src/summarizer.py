from __future__ import annotations

from typing import Dict, List


def _first_n_sentences(text: str, n: int) -> str:
    parts = [p.strip() for p in text.split(".") if p.strip()]
    return ". ".join(parts[:n]) + ("." if parts else "")


def generate_summaries(case_text: str, top_sections: List[Dict[str, str | float]]) -> Dict[str, str]:
    section_line = ", ".join([str(item["statute_id"]) for item in top_sections[:5]])

    legal_summary = (
        "Based on OCR-extracted case material, likely statutory anchors include "
        f"{section_line}. The text indicates allegations and judicial reasoning that align "
        "with these sections. Final mapping is produced through combined regex citation evidence "
        "and semantic section similarity, followed by graph-based probability ranking. "
        f"Key excerpt: {_first_n_sentences(case_text, 4)}"
    )

    layman_summary = (
        "This document appears to describe a legal dispute where the important law sections are "
        f"{section_line}. We first read the PDF with OCR, then look for explicit section mentions, "
        "and then compare the full text against known law descriptions. "
        f"In short: {_first_n_sentences(case_text, 3)}"
    )

    return {
        "legal_summary": legal_summary,
        "layman_summary": layman_summary,
    }
