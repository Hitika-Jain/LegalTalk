from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GraphResult:
    statute_id: str
    probability: float


def graph_based_rerank(
    regex_probs: Dict[str, float],
    bert_probs: Dict[str, float],
    top_k: int = 10,
) -> List[GraphResult]:
    """
    Graph-style fusion: statutes are nodes, and probability is weighted vote from
    regex (explicit citations) and semantic mapper (contextual similarity).
    """

    all_nodes = set(regex_probs) | set(bert_probs)
    fused: Dict[str, float] = {}
    for node in all_nodes:
        fused[node] = (0.6 * regex_probs.get(node, 0.0)) + (0.4 * bert_probs.get(node, 0.0))

    if not fused:
        return []

    max_score = max(fused.values()) or 1.0
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [GraphResult(statute_id=k, probability=(v / max_score) * 100.0) for k, v in ranked]
