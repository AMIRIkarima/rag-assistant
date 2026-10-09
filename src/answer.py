import os
import time
from dotenv import load_dotenv
from mistralai.client import Mistral

from src.retrieve import retrieve
from src.config import LLM_MODEL

load_dotenv()  # reads .env
llm = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

REFUSAL = "I cannot answer this question based on the provided excerpts."
SYSTEM = (
    "Answer using only the provided excerpts. Do not use outside knowledge. "
    "Cite every factual claim with the matching [n] reference. "
    f"If the excerpts do not contain enough information to answer, respond exactly: {REFUSAL}"
)

def generate(system: str, user: str) -> str:
    for attempt in range(3):
        try:
            resp = llm.chat.complete(
                model=LLM_MODEL,
                max_tokens=800,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return resp.choices[0].message.content
        except Exception:                 # rate limit on the free tier
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))

def answer(q):
    hits = retrieve(q)
    context = "\n\n".join(f"[{i+1}] ({m['source']}, p.{m['page']})\n{d}"
                          for i, (d, m) in enumerate(hits))
    text = generate(SYSTEM, f"Excerpts:\n{context}\n\nQuestion: {q}")
    sources = [{"ref": i + 1, **metadata} for i, (_, metadata) in enumerate(hits)]
    return {"answer": text, "sources": sources}