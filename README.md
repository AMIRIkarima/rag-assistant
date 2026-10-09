# Scientific Paper RAG Assistant

A research assistant for papers about gravitational waves. The project uses PyMuPDF for text extraction, Sentence Transformers for local embeddings, ChromaDB for vector search, Mistral for answer generation, and FastAPI for the API.

## Environment

Python 3.12 or newer is recommended. From the project root, create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select `.venv\Scripts\python.exe` as the interpreter. The first ingestion run downloads the `BAAI/bge-small-en-v1.5` embedding model.

## Corpus and indexing

To download up to 25 arXiv papers matching the default query and index their pages:

```powershell
python -m src.ingest --download --max-results 25
```

Ingestion saves PDFs to `data/pdfs/`, stores their titles and arXiv IDs in `data/papers.json`, and creates the persistent index in `chroma_db/`. Downloaded PDFs and the index are ignored by Git. To index your own PDFs, place them in `data/pdfs/` and run `python -m src.ingest`. Filenames are used as source IDs when arXiv metadata is unavailable.

The default chunk size is 1,000 characters with 200 characters of overlap. Chunking options are also available in `ingest_papers()` for controlled comparisons.

## Questions and API

Vector search is available through `retrieve(question, k=5)` in `src.retrieve`. To generate a cited answer, configure your API key without adding it to the repository:

```powershell
$env:MISTRAL_API_KEY = "..."
```

The default generation model is `open-mistral-nemo` (configured as `LLM_MODEL` in `src/config.py`).

Start the API:

```powershell
uvicorn src.api:app --reload
```

Try `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/ask?q=...`. The response contains the generated answer and metadata for its source excerpts. The model is instructed to cite claims using `[n]` and explicitly state when information is missing.

## Retrieval evaluation

Add at least 20 questions with verified answers and the expected arXiv ID to `eval/questions.jsonl` (one JSON object per line):

```json
{"question":"What is LIGO's primary scientific goal?","source":"0711.3041"}
```

Run `python -m eval.run_eval` to report hit@k, its 95% Wilson confidence interval, and MRR for k = 1, 3, 5, and 10. The included questions are a starting point, not a sufficiently large evaluation set for drawing conclusions. Evaluate multiple chunk sizes and embedding models separately; record results and failures after manual review.

## Docker

Build the index locally first, then build and run the image from the project root:

```powershell
docker build -t rag-app .
docker run --rm -e MISTRAL_API_KEY -p 8000:8000 rag-app
```

Never put the Mistral API key in the image or repository.
