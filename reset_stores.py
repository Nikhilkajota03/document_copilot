"""
reset_stores.py
---------------
Wipes all vector and document stores so a new document can be ingested cleanly.

Clears:
  - data/chroma_db/     (ChromaDB dense vector index)
  - data/doc_store/     (LocalFileStore raw parent chunks)
  - data/bm25_index.pkl (BM25 sparse keyword index)

Usage:
  uv run python reset_stores.py
"""

import shutil
import os

STORES = {
    "ChromaDB (dense vectors)":   "data/chroma_db",
    "Doc Store (raw chunks)":     "data/doc_store",
    "BM25 Index (sparse index)":  "data/bm25_index.pkl",
}

def reset():
    print("=== Document Copilot — Store Reset ===\n")
    
    for label, path in STORES.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
                os.makedirs(path)  # Recreate empty dir so pipeline doesn't error
                print(f"  ✓ Cleared:  {label}  ({path}/)")
            else:
                os.remove(path)
                print(f"  ✓ Deleted:  {label}  ({path})")
        else:
            print(f"  - Skipped:  {label}  (not found, already clean)")
    
    print("\n✅ All stores reset. You can now run the ingestion pipeline on a new document.")
    print("   → uv run python -m app.ingestion.pipeline\n")

if __name__ == "__main__":
    reset()
