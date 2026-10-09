from fastapi import FastAPI, HTTPException, Query

from src.answer import answer

app = FastAPI(title="Scientific Paper RAG Assistant")


@app.get("/ask")
def ask(q: str = Query(min_length=1, description="Question about the indexed papers")) -> dict:
    try:
        return answer(q)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error