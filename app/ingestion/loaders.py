import os
import uuid
import pickle
import time
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage.encoder_backed import EncoderBackedStore
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

load_dotenv()

# We use BAAI/bge-small-en-v1.5 for embeddings (lightweight and fast)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def load_documents_multi_vector(
    docs: List[Document], 
    vectorstore_path: str = "data/chroma_db", 
    docstore_path: str = "data/doc_store",
    bm25_path: str = "data/bm25_index.pkl"
) -> tuple[MultiVectorRetriever, BM25Retriever]:
    """
    Summarizes document chunks and loads them into a Multi-Vector Retriever setup.
    """
    print(f"Setting up Vector Database at '{vectorstore_path}' and DocStore at '{docstore_path}'...")
    
    # 1. Initialize Embeddings and Storage
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vectorstore = Chroma(
        collection_name="multi_vector_docs",
        embedding_function=embeddings,
        persist_directory=vectorstore_path
    )
    
    # The Document Store needs to be backed by a disk file store but encoded to handle LangChain Document objects
    fs_store = LocalFileStore(docstore_path)
    store = EncoderBackedStore(
        store=fs_store,
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads
    )
    
    # The retriever that ties the two together
    id_key = "parent_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )
    
    # 2. Setup the Summarization LLM
    print("Initializing Groq LLM for summarization...")
    llm = ChatGroq(model_name="groq/compound")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant tasked with summarizing text and tables. Give a concise summary of the following content, specifically ensuring that any complex structures or data tables are explained in clear natural language so they can be easily found via semantic search."),
        ("human", "{element}")
    ])
    
    summarize_chain = prompt | llm | StrOutputParser()
    
    # 3. Process documents
    print(f"Summarizing {len(docs)} chunks... (This may take a minute due to API limits)")
    doc_ids = [str(uuid.uuid4()) for _ in docs]
    
    summary_docs = []
    
    for i, doc in enumerate(docs):
        # Generate Summary with retry and backoff
        retries = 3
        summary_text = ""
        while retries > 0:
            try:
                summary_text = summarize_chain.invoke({"element": doc.page_content})
                time.sleep(1.5) # Buffer between normal requests to respect TPM/RPM
                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    print(f"  Rate limit hit at chunk {i}. Sleeping for 10s...")
                    time.sleep(10)
                    retries -= 1
                else:
                    print(f"Warning: Failed to summarize chunk {i}. Error: {e}")
                    break
                    
        if not summary_text:
            summary_text = doc.page_content[:200] # Fallback to raw text
            
        # Create the child document (Summary)
        summary_doc = Document(
            page_content=summary_text,
            metadata={id_key: doc_ids[i], **doc.metadata} # Carry over headers/source metadata
        )
        summary_docs.append(summary_doc)
        
        if (i+1) % 5 == 0:
            print(f"  ...Summarized {i+1}/{len(docs)} chunks")
            
    # 4. Load into DBs
    print(f"Loading {len(summary_docs)} first it convert to vector then save")
    retriever.vectorstore.add_documents(summary_docs)
    
    print(f"Loading {len(docs)} raw documents into Key-Value Store...")
    retriever.docstore.mset(list(zip(doc_ids, docs)))
    
    # 5. Build and Save BM25 Index
    print("Building BM25 Sparse Index on raw markdown chunks...")
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 4
    
    print(f"Saving BM25 index to {bm25_path}...")
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    
    print("Ingestion complete!")
    return retriever, bm25_retriever
