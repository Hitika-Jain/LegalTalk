from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from src.bert_mapper import BertSectionMapper, bert_probability_map
from src.graph_mapper import graph_based_rerank
from src.regex_mapper import map_statutes_with_regex, regex_probability_map
from src.summarizer import generate_summaries


class StatutePipeline:
    def __init__(self, statutes_csv_path: str) -> None:
        self.bert_mapper = BertSectionMapper(statutes_csv_path)

    def run(self, case_text: str) -> Dict[str, Any]:
        regex_hits = map_statutes_with_regex(case_text)
        regex_probs = regex_probability_map(regex_hits)

        bert_matches = self.bert_mapper.map_sections(case_text)
        bert_probs = bert_probability_map(bert_matches)

        top10 = graph_based_rerank(regex_probs, bert_probs, top_k=10)
        top10_dict = [asdict(item) for item in top10]

        summaries = generate_summaries(case_text, top10_dict)

        return {
            "regex_hits": [asdict(hit) for hit in regex_hits],
            "bert_matches": [asdict(match) for match in bert_matches],
            "top_sections": top10_dict,
            "summaries": summaries,
        }
