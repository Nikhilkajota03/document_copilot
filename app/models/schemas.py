from pydantic import BaseModel
from typing import Optional


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class SourceChunk(BaseModel):
    content: str
    metadata: dict


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
