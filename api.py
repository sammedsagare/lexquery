from fastapi import FastAPI
from pydantic import BaseModel
from main import get_similar_chunks, get_response_from_query, memory

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    context, sources = get_similar_chunks(request.question)
    answer = get_response_from_query(request.question, context, memory)
    return QueryResponse(answer=answer, sources=sources)