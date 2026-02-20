from __future__ import annotations

import sys
from pathlib import Path

# Add the project root to the path so imports work correctly
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import streamlit as st

from src.preprocessing import parse_uploaded_file
from src.statute_ensemble import StatutePipeline


@st.cache_resource
def get_pipeline() -> StatutePipeline:
    data_path = Path(__file__).resolve().parent / "data" / "statutes.csv"
    return StatutePipeline(str(data_path))


def main() -> None:
    st.title("Legal PDF Analyzer")
    st.write(
        "Upload a PDF/image. The app runs OCR/text extraction, regex statute mapping, "
        "semantic section mapping, graph-based fusion, and then creates legal + layman summaries."
    )

    uploaded = st.file_uploader("Upload case file", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"])
    if not uploaded:
        st.info("Please upload a file.")
        return

    try:
        parsed = parse_uploaded_file(uploaded.name, uploaded.getvalue())
    except Exception as exc:
        st.error(f"Unable to parse file: {exc}")
        return

    pipeline = get_pipeline()
    result = pipeline.run(parsed.text)

    st.subheader("Top 10 most probable IPC sections")
    if result["top_sections"]:
        st.dataframe(result["top_sections"], use_container_width=True)
    else:
        st.warning("No likely section was mapped.")

    st.subheader("Layman summary")
    st.write(result["summaries"]["layman_summary"])

    st.subheader("Legal summary")
    st.write(result["summaries"]["legal_summary"])

    with st.expander("Debug details"):
        st.markdown("**Regex mapper hits**")
        st.json(result["regex_hits"])
        st.markdown("**Semantic mapper hits (BERT-style)**")
        st.json(result["bert_matches"][:20])
        st.markdown("**Extracted text preview**")
        st.write(parsed.text[:4000])


if __name__ == "__main__":
    main()
