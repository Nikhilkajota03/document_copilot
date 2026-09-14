from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import chat, documents

app = FastAPI(
    title="Document Copilot API",
    description="RAG-powered Q&A API for financial documents",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "Document Copilot API is running"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
