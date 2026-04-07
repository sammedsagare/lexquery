from fastapi import FastAPI
from pydantic import BaseModel, Field

from main import (
    get_response_from_query,
    get_similar_chunks,
    memory,
)
from langchain_core.chat_history import InMemoryChatMessageHistory

app = FastAPI(title="LexQuery API")

session_memories: dict[str, InMemoryChatMessageHistory] = {}


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


def get_session_memory(session_id: str | None) -> InMemoryChatMessageHistory:
    if not session_id:
        return memory
    if session_id not in session_memories:
        session_memories[session_id] = InMemoryChatMessageHistory()
    return session_memories[session_id]


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest) -> QueryResponse:
    chat_memory = get_session_memory(request.session_id)
    context, sources = get_similar_chunks(request.question)
    answer = get_response_from_query(request.question, context, chat_memory)
    return QueryResponse(answer=answer, sources=sources)
