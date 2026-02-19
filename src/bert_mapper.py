from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set

import pandas as pd


@dataclass
class BertMatch:
    statute_id: str
    section_title: str
    score: float


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "is", "for", "with", "under", "on"}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text.lower()) if t.lower() not in _STOP]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class BertSectionMapper:
    """Offline semantic mapper using token-overlap proxy for BERT scores."""

    def __init__(self, statutes_csv_path: str) -> None:
        self.df = pd.read_csv(statutes_csv_path).fillna("")
        self.df["lookup_text"] = (
            self.df["Offense"].astype(str)
            + " "
            + self.df["Description"].astype(str)
            + " "
            + self.df["id"].astype(str)
        )
        self.df["token_set"] = self.df["lookup_text"].apply(lambda x: set(_tokenize(x)))

    def map_sections(self, case_text: str, top_k: int = 20) -> List[BertMatch]:
        query_tokens = set(_tokenize(case_text))
        scored = []
        for _, row in self.df.iterrows():
            score = _jaccard(query_tokens, row["token_set"])
            if score > 0:
                scored.append(
                    BertMatch(
                        statute_id=row["id"],
                        section_title=row["Offense"],
                        score=float(score),
                    )
                )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


def bert_probability_map(matches: List[BertMatch]) -> Dict[str, float]:
    if not matches:
        return {}
    max_score = max(m.score for m in matches)
    if max_score <= 0:
        return {}
    return {m.statute_id: m.score / max_score for m in matches}
