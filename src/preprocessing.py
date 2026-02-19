from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pdfplumber
import pytesseract
from PIL import Image


@dataclass
class DocumentText:
    text: str
    sentences: List[str]


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text)
    return [c.strip() for c in chunks if c.strip()]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text with embedded-text first and OCR as fallback per page image."""
    text_parts: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = (page.extract_text() or "").strip()
            if page_text:
                text_parts.append(page_text)
                continue

            page_image = page.to_image(resolution=200).original
            ocr_text = pytesseract.image_to_string(page_image)
            if ocr_text.strip():
                text_parts.append(ocr_text)

    return _clean_text("\n".join(text_parts))


def extract_text_from_image(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes))
    return _clean_text(pytesseract.image_to_string(image))


def parse_uploaded_file(file_name: str, file_bytes: bytes) -> DocumentText:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".pdf":
        text = extract_text_from_pdf(file_bytes)
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        text = extract_text_from_image(file_bytes)
    else:
        raise ValueError("Only PDF and image files are supported in this workflow.")

    if not text:
        raise ValueError("No readable text found in the uploaded file.")

    return DocumentText(text=text, sentences=_split_sentences(text))
