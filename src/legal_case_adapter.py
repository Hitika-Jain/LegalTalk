"""
legal_case_adapter.py
---------------------
Canonical schema + validator for the ensemble (regex + LegalBERT + graph) output,
plus an adapter that converts it into the flat arguments your `run_case` expects.

Usage
-----
from legal_case_adapter import EnsembleCase, to_run_case_args, run_case_with_ensemble

# If you already have a dict from your ensemble:
ens_dict = {...}
_ = run_case_with_ensemble(run_case, ens_dict)

# Or validate first:
ens = EnsembleCase.model_validate(ens_dict)
args = to_run_case_args(ens)
_ = run_case(*args[:-1], label=args[-1])

Notes
-----
- Keeps spans/confidences in the schema for audit, strips them when passing to run_case.
- Robust to partially missing sections (returns empty lists where needed).
- Includes a very small `run_case` stub for local testing if your real function isn't imported.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple, Union
from pydantic import BaseModel, Field, validator


# -----------------------------
# Schema
# -----------------------------

class Statute(BaseModel):
    code: str = Field(..., description="e.g., 'Section 420 IPC'")
    title: Optional[str] = Field(None, description="Short title like 'cheating'")
    cited_text: Optional[str] = Field(None, description="Optional verbatim mention from text")
    confidence: Optional[float] = Field(None, ge=0, le=1)


class SentenceItem(BaseModel):
    text: str
    span: Optional[List[int]] = Field(None, description="[start, end] character offsets")
    confidence: Optional[float] = Field(None, ge=0, le=1)


class Relation(BaseModel):
    head: str
    relation: str
    tail: str
    evidence_spans: Optional[List[List[int]]] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)


class Parties(BaseModel):
    appellant: Optional[str] = None
    respondent: Optional[str] = None


class Metadata(BaseModel):
    label: Optional[str] = None
    court: Optional[str] = None
    bench: Optional[List[str]] = None
    decision_date: Optional[str] = None
    citation: Optional[str] = None
    case_no: Optional[str] = None
    parties: Optional[Parties] = None
    source_file: Optional[str] = None


class Arguments(BaseModel):
    Appellant: List[SentenceItem] = Field(default_factory=list)
    Respondent: List[SentenceItem] = Field(default_factory=list)
    Other: List[SentenceItem] = Field(default_factory=list)


class EnsembleCase(BaseModel):
    case_text: str = Field(..., description="Full raw judgment text")
    statutes: List[Statute] = Field(default_factory=list)
    facts: List[SentenceItem] = Field(default_factory=list)
    issues: List[SentenceItem] = Field(default_factory=list)
    arguments: Arguments = Field(default_factory=Arguments)
    reasoning: List[SentenceItem] = Field(default_factory=list)
    order: List[SentenceItem] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)

    @validator("case_text")
    def case_text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("case_text must be non-empty")
        return v


# -----------------------------
# Adapter to run_case()
# -----------------------------

def _fmt_statute(s: Statute) -> str:
    if s.title:
        return f"{s.code} ({s.title})"
    return s.code

def to_run_case_args(ens: Union[EnsembleCase, Dict[str, Any]]
                     ) -> Tuple[str, List[str], List[str], List[str], List[str], List[str], List[str], Optional[str]]:
    """
    Convert EnsembleCase → positional args for run_case().
    Returns: (case_text, statutes, facts_sents, issues_sents,
              arguments_sents, reasoning_sents, order_sents, label)
    """
    if not isinstance(ens, EnsembleCase):
        ens = EnsembleCase.model_validate(ens)

    case_text = ens.case_text

    statutes = [_fmt_statute(s) for s in (ens.statutes or [])]

    facts_sents = [x.text for x in (ens.facts or [])]
    issues_sents = [x.text for x in (ens.issues or [])]

    arguments_sents: List[str] = []
    for party_name, items in (("Appellant", ens.arguments.Appellant),
                              ("Respondent", ens.arguments.Respondent),
                              ("Other", ens.arguments.Other)):
        for it in items or []:
            if it and it.text:
                arguments_sents.append(f"{party_name}: {it.text}")

    reasoning_sents = [x.text for x in (ens.reasoning or [])]
    order_sents = [x.text for x in (ens.order or [])]

    label = ens.metadata.label if ens.metadata else None
    return (case_text, statutes, facts_sents, issues_sents,
            arguments_sents, reasoning_sents, order_sents, label)


# -----------------------------
# Convenience wrapper
# -----------------------------

def run_case_with_ensemble(run_case_fn, ens: Union[EnsembleCase, Dict[str, Any]]):
    """
    Call your existing run_case() with a validated ensemble object.
    """
    (case_text, statutes, facts_sents, issues_sents,
     arguments_sents, reasoning_sents, order_sents, label) = to_run_case_args(ens)

    # run_case signature: run_case(case_text, statutes, facts_sents, issues_sents, arguments_sents, reasoning_sents, order_sents, label=?)
    if label is not None:
        return run_case_fn(case_text, statutes, facts_sents, issues_sents,
                           arguments_sents, reasoning_sents, order_sents, label=label)
    else:
        return run_case_fn(case_text, statutes, facts_sents, issues_sents,
                           arguments_sents, reasoning_sents, order_sents)


# -----------------------------
# Optional: local testing stub
# -----------------------------

def _run_case_stub(case_text, statutes, facts_sents, issues_sents, arguments_sents, reasoning_sents, order_sents, label=None):
    """A tiny stand-in for quick manual tests; replace with your real run_case."""
    return {
        "label": label,
        "lengths": {
            "case_text": len(case_text),
            "statutes": len(statutes),
            "facts": len(facts_sents),
            "issues": len(issues_sents),
            "arguments": len(arguments_sents),
            "reasoning": len(reasoning_sents),
            "order": len(order_sents),
        },
        "preview": {
            "facts_head": facts_sents[:2],
            "issues_head": issues_sents[:2],
            "order_head": order_sents[:1]
        }
    }


if __name__ == "__main__":
    # Minimal smoke test with your toy example
    example = {
        "case_text": "IN THE SUPREME COURT OF INDIA ... full text ...",
        "statutes": [
            {"code": "Section 420 IPC", "title": "cheating", "confidence": 0.97},
            {"code": "Section 415 IPC", "title": "definition of cheating", "confidence": 0.91}
        ],
        "facts": [
            {"text": "Complainant transferred ₹12,00,000 based on the accused's representation about property rights.", "confidence": 0.93},
            {"text": "No registered sale deed was executed despite repeated requests.", "confidence": 0.90}
        ],
        "issues": [
            {"text": "Whether the ingredients of Section 420 IPC are made out.", "confidence": 0.88}
        ],
        "arguments": {
            "Appellant": [{"text": "The dispute is civil in nature; at most a breach of contract."}],
            "Respondent": [{"text": "Dishonest inducement existed at the inception of the transaction."}],
            "Other": []
        },
        "reasoning": [
            {"text": "Fraudulent intent at inception is essential under Section 420 IPC."},
            {"text": "Evidence indicates the accused knew he lacked title when inducing payment."}
        ],
        "order": [
            {"text": "Appeal dismissed; conviction under Section 420 IPC upheld; sentence maintained."}
        ],
        "metadata": { "label": "Toy IPC 420" }
    }

    args = to_run_case_args(example)
    print("Args prepared (truncated preview):")
    print("label:", args[-1])
    print("statutes:", args[1])
    print("facts:", args[2][:1])
    out = _run_case_stub(*args[:-1], label=args[-1])
    print("Stub output:", out)
