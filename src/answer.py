import os
import time
from dotenv import load_dotenv
from mistralai.client import Mistral

from src.retrieve import retrieve
from src.retrieve import retrieve

load_dotenv()  # reads .env
llm = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

SYSTEM = ("Answer only from the provided excerpts. Cite every claim with [n]. "
          "If the excerpts are not sufficient, say so explicitly.")

def generate(system: str, user: str) -> str:
    for attempt in range(3):
        try:
            resp = llm.chat.complete(
                model="mistral-small-latest",
                max_tokens=800,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return resp.choices[0].message.content
        except Exception as e:           # rate limit on the free tier
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))

def answer(q):
    hits = retrieve(q)
    context = "\n\n".join(f"[{i+1}] ({m['source']}, p.{m['page']})\n{d}"
                          for i, (d, m) in enumerate(hits))
    text = generate(SYSTEM, f"Excerpts:\n{context}\n\nQuestion: {q}")
    return {"answer": text, "sources": [m for _, m in hits]}