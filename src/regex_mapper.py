from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from src.utils import normalize_to_ipc


@dataclass
class RegexHit:
    statute_id: str
    mention: str
    count: int


_SECTION_PATTERN = re.compile(
    r"\b(?:section|sections|sec|s\.?|§)\s*([0-9]{1,3}[A-Za-z]?(?:\([0-9A-Za-z]+\))?(?:[-][A-Za-z])?)",
    re.IGNORECASE,
)

_IPC_PATTERN = re.compile(
    r"\b([0-9]{1,3}[A-Za-z]?(?:\([0-9A-Za-z]+\))?(?:[-][A-Za-z])?)\s*(?:ipc|indian\s+penal\s+code)\b",
    re.IGNORECASE,
)


def map_statutes_with_regex(text: str) -> List[RegexHit]:
    counts: Dict[str, Dict[str, int | str]] = {}

    for pattern in (_SECTION_PATTERN, _IPC_PATTERN):
        for match in pattern.finditer(text):
            raw = match.group(1)
            canonical = normalize_to_ipc(raw)
            if not canonical:
                continue
            if canonical not in counts:
                counts[canonical] = {"count": 0, "mention": raw}
            counts[canonical]["count"] += 1

    ranked = sorted(counts.items(), key=lambda kv: kv[1]["count"], reverse=True)
    return [
        RegexHit(statute_id=statute, mention=meta["mention"], count=int(meta["count"]))
        for statute, meta in ranked
    ]


def regex_probability_map(hits: List[RegexHit]) -> Dict[str, float]:
    if not hits:
        return {}
    max_count = max(hit.count for hit in hits)
    if max_count == 0:
        return {}
    return {hit.statute_id: hit.count / max_count for hit in hits}
