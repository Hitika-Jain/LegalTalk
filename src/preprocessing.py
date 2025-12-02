import fitz  # PyMuPDF
import json
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import os

def parse_pdf_to_sentences(pdf_path, out_path=None):
    """
    Extracts text page-by-page, splits into sentences,
    and keeps (page, start_char, end_char) offsets.
    """
    doc = fitz.open(pdf_path)
    all_records = []

    for pno in range(len(doc)):
        page = doc.load_page(pno)
        text = page.get_text("text")

        # sentence tokenize
        sentences = sent_tokenize(text)

        cursor = 0
        for i, sent in enumerate(sentences):
            idx = text.find(sent, cursor)
            if idx == -1:
                idx = cursor  # fallback

            record = {
                "page": pno,
                "line_id": i,
                "start_char": idx,
                "end_char": idx + len(sent),
                "text": sent
            }
            all_records.append(record)
            cursor = idx + len(sent)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[Saved] {len(all_records)} lines → {out_path}")

    return all_records

# Example usage in Colab:
# parse_pdf_to_sentences("/content/drive/MyDrive/legal_papers/sample.pdf",
#                        "/content/drive/MyDrive/parsed/sample.jsonl")
