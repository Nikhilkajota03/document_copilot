import os
from .parser import parse_document
from .splitter import split_markdown_documents
from .loaders import load_documents_multi_vector

from langchain_classic.retrievers import EnsembleRetriever


def run_ingestion_pipeline(pdf_path: str) -> dict:
    """
    Runs the full ingestion pipeline for a given PDF path.
    Returns a summary dict with status and chunk count.
    """
    print("=== Starting Ingestion Pipeline ===")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Could not find the PDF at: {pdf_path}")

    # 1. Parsing
    print("\n--- STEP 1: PARSING ---")
    raw_docs = parse_document(pdf_path)
    if not raw_docs:
        raise RuntimeError("Parsing failed — no documents returned.")

    # 2. Splitting
    print("\n--- STEP 2: SPLITTING ---")
    chunks = split_markdown_documents(raw_docs)

    # 3. Summarization and DB Loading
    print("\n--- STEP 3: SUMMARIZING & LOADING ---")
    multi_vector_retriever, bm25_retriever = load_documents_multi_vector(chunks)

    # 4. Hybrid Search Setup
    print("\n--- STEP 4: HYBRID SEARCH SETUP ---")
    EnsembleRetriever(
        retrievers=[bm25_retriever, multi_vector_retriever],
        weights=[0.5, 0.5],
    )

    print("\n=== Ingestion Pipeline Complete! ===")
    return {
        "status": "success",
        "file": pdf_path,
        "chunks_ingested": len(chunks),
    }


def main():
    pdf_path = "data/uploads/TSLA-Q4-2025-Update.pdf"
    result = run_ingestion_pipeline(pdf_path)
    print(result)

    # Quick test search
    from app.retrieval.retriever import get_ensemble_retriever
    retriever = get_ensemble_retriever()
    query = "TSLA Inc."
    print(f"Searching for: '{query}'")
    results = retriever.invoke(query)
    if results:
        print("Found a match! Here is the original raw content:")
        print(results[0].page_content[:300])
    else:
        print("No results found.")


if __name__ == "__main__":
    main()
