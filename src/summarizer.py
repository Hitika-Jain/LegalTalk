from __future__ import annotations

import re
from typing import Dict, List

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
except ImportError:
    sent_tokenize = None


def _download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)


def _tokenize_sentences(text: str) -> List[str]:
    """Tokenize text into sentences."""
    if sent_tokenize:
        try:
            _download_nltk_data()
            return sent_tokenize(text)
        except:
            pass
    # Fallback: simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _extract_key_sentences(text: str, n: int = 3) -> List[str]:
    """Extract n most important sentences from text."""
    sentences = _tokenize_sentences(text)
    if len(sentences) <= n:
        return sentences
    
    # Score sentences by length and keyword presence
    scored = []
    keywords = ['held', 'court', 'section', 'appeal', 'conviction', 'dismissed', 'upheld',
                'guilty', 'not guilty', 'evidence', 'judgment', 'order', 'case', 'accused',
                'parties', 'facts', 'law', 'legal', 'proved', 'entitled']
    
    for sent in sentences:
        sent_clean = sent.strip()
        if len(sent_clean) < 20:  # Skip very short sentences
            continue
            
        score = len(sent_clean.split())  # Base score on length
        score += sum(5 for kw in keywords if kw.lower() in sent_clean.lower())  # Boost for keywords
        scored.append((sent_clean, score))
    
    if not scored:
        return sentences[:n]
    
    # Sort by score and return top n
    scored.sort(key=lambda x: x[1], reverse=True)
    top_sents = [s[0] for s in scored[:n]]
    return [s.capitalize() for s in top_sents]


def _extract_statute_descriptions(top_sections: List[Dict[str, str | float]]) -> str:
    """Create descriptions for top statutes."""
    if not top_sections:
        return "relevant legal provisions"
    
    sections = []
    for item in top_sections[:3]:
        statute_id = item.get("statute_id", "")
        title = item.get("section_title", "")
        if statute_id:
            if title:
                sections.append(f"{statute_id} ({title})")
            else:
                sections.append(statute_id)
    
    return ", ".join(sections) if sections else "relevant legal provisions"


def generate_summaries(case_text: str, top_sections: List[Dict[str, str | float]]) -> Dict[str, str]:
    """Generate both legal and layman summaries from case text and relevant sections."""
    
    # Extract key information
    key_sentences = _extract_key_sentences(case_text, n=2)
    statute_info = _extract_statute_descriptions(top_sections)
    
    # Get first sentence for context
    all_sentences = _tokenize_sentences(case_text)
    first_sentence = all_sentences[0] if all_sentences else "A legal matter was presented before the court."
    
    # Build context from key sentences
    key_context = " ".join(key_sentences[:2]) if key_sentences else _first_n_sentences(case_text, 2)
    
    # Clean up first sentence if it's too long
    if len(first_sentence) > 150:
        first_sentence = first_sentence[:147] + "..."
    
    # Generate legal summary (professional, technical language)
    legal_summary = (
        f"This matter involves the statutory provisions of {statute_info}. "
        f"The factual background: {first_sentence} "
        f"Application of law: {key_context} "
        f"The court's determination considers the requisite elements and evidentiary standards under the applicable legal provisions."
    )
    
    # Generate layman summary (simple, easy to understand)
    layman_summary = (
        f"The important law(s) in this case: {statute_info}. "
        f"What happened: {first_sentence} "
        f"What the court found: {key_context} "
        f"Bottom line: The court made a judgment based on the facts and the applicable laws."
    )
    
    return {
        "legal_summary": legal_summary,
        "layman_summary": layman_summary,
    }


def _first_n_sentences(text: str, n: int) -> str:
    """Fallback function to get first n sentences."""
    parts = [p.strip() for p in text.split(".") if p.strip()]
    return ". ".join(parts[:n]) + ("." if parts else "")
