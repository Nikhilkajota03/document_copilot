import os
import pickle
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage.encoder_backed import EncoderBackedStore
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_vector_store(vectorstore_path: str = "data/chroma_db"):
    return Chroma(
        collection_name="multi_vector_docs",
        embedding_function=get_embeddings(),
        persist_directory=vectorstore_path
    )

def get_doc_store(docstore_path: str = "data/doc_store"):
    fs_store = LocalFileStore(docstore_path)
    return EncoderBackedStore(
        store=fs_store,
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads
    )
