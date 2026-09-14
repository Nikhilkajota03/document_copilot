import pickle
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from .vector_store import get_vector_store, get_doc_store

def get_multi_vector_retriever(vectorstore_path: str = "data/chroma_db", docstore_path: str = "data/doc_store"):
    """
    Reconstructs the Multi-Vector Retriever that maps semantic summaries 
    back to their original raw chunks in the DocStore.
    """
    vectorstore = get_vector_store(vectorstore_path)
    docstore = get_doc_store(docstore_path)
    
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=docstore,
        id_key="parent_id",
    )

def get_bm25_retriever(bm25_path: str = "data/bm25_index.pkl"):
    """
    Loads the saved BM25 sparse index for exact keyword matching.
    """
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)
    return bm25_retriever

def get_ensemble_retriever(weights: list[float] = [0.5, 0.5]):
    """
    Combines the Multi-Vector Retriever (Semantic Search) and 
    BM25 Retriever (Keyword Search) using Reciprocal Rank Fusion (RRF).
    """
    multi_vector_retriever = get_multi_vector_retriever()
    bm25_retriever = get_bm25_retriever()
    
    # *** RRF is applied here under the hood by esembly retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[multi_vector_retriever, bm25_retriever],
        weights=weights
    )
    return ensemble_retriever

def get_reranked_retriever(weights: list[float] = [0.5, 0.5], top_n: int = 3):
    """
    Wraps the EnsembleRetriever with a Cross-Encoder Reranker.
    It takes the top results from RRF and scores them against the query
    to find the absolute most relevant chunks.
    """
    base_retriever = get_ensemble_retriever(weights=weights)
    
    # Initialize a fast, accurate cross-encoder model
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Setup the reranker to output the top 3 results
    compressor = CrossEncoderReranker(model=model, top_n=top_n)
    
    # Wrap the base ensemble retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever
