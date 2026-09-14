import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from app.retrieval.retriever import get_reranked_retriever
from app.models.schemas import SourceChunk
from dotenv import load_dotenv

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain():
    """
    Sets up the full Multimodal RAG Generation Pipeline.
    Retrieves the raw uncorrupted chunks (via Ensemble Retriever) and passes them to the LLM.
    """
    # 1. Initialize Retriever
    retriever = get_reranked_retriever(weights=[0.5, 0.5])

    # 2. Initialize LLM for Generation
    llm = ChatGroq(model_name="groq/compound")

    # 3. Create the Prompt
    template = """You are an expert financial analyst answering questions about complex documents like 10-K reports.
Use the following pieces of retrieved context to answer the question. 
The context may contain raw Markdown tables and text paragraphs.
If you don't know the answer based on the context, just say that you don't know. 
Keep the answer concise but provide specific data points if they are in the context.

Context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Build the Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def get_rag_chain_with_sources():
    """
    Returns both the final answer and the source documents used for generation.
    """
    retriever = get_reranked_retriever(weights=[0.5, 0.5])
    llm = ChatGroq(model_name="openai/gpt-oss-120b")

    template = """You are an expert financial analyst answering questions about complex documents like 10-K reports.
Use the following pieces of retrieved context to answer the question. 
The context may contain raw Markdown tables and text paragraphs.
If you don't know the answer based on the context, just say that you don't know. 
Keep the answer concise but provide specific data points if they are in the context.

Context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    def run_with_sources(question: str) -> dict:
        docs = retriever.invoke(question)
        context = format_docs(docs)
        answer = (prompt | llm | StrOutputParser()).invoke(
            {"context": context, "question": question}
        )
        sources = [
            SourceChunk(content=doc.page_content, metadata=doc.metadata)
            for doc in docs
        ]
        return {"answer": answer, "sources": sources}

    return run_with_sources


async def stream_rag_answer(question: str):
    """
    Async generator that streams tokens from the RAG chain for SSE.
    """
    retriever = get_reranked_retriever(weights=[0.5, 0.5])
    llm = ChatGroq(model_name="openai/gpt-oss-120b")

    template = """You are an expert financial analyst answering questions about complex documents like 10-K reports.
Use the following pieces of retrieved context to answer the question. 
The context may contain raw Markdown tables and text paragraphs.
If you don't know the answer based on the context, just say that you don't know. 
Keep the answer concise but provide specific data points if they are in the context.

Context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Retrieve docs in a thread (sync call)
    docs = await asyncio.to_thread(retriever.invoke, question)
    context = format_docs(docs)

    # Stream the LLM response
    chain = prompt | llm | StrOutputParser()
    async for token in chain.astream({"context": context, "question": question}):
        yield token