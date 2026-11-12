# Scientific Paper RAG Assistant
 
A research assistant for papers about gravitational waves. The project uses PyMuPDF for text extraction, Sentence Transformers for local embeddings, ChromaDB for vector search, Mistral for answer generation, and FastAPI for the API.
 
The project is built and managed with [uv](https://docs.astral.sh/uv/): dependencies, the virtual environment, the version number and the packaging (wheel and source archive) are all driven by `pyproject.toml` and `uv.lock`.
 
## Quick start
 
```powershell
git clone git@github.com:AMIRIkarima/rag-assistant.git
cd rag-assistant
uv sync
```
 
`uv sync` creates `.venv`, installs the exact dependency versions recorded in `uv.lock` and installs the project itself. No manual `pip` or `venv` commands are needed.
 
## Requirements
 
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (check with `uv --version`).
- Python 3.12 to 3.14. The project pins Python 3.12 in `.python-version`, and uv downloads it automatically if it is missing. Python 3.15 is excluded because `onnxruntime` (used by ChromaDB) has no wheels for it yet.
- A Mistral API key to generate answers (not needed for indexing and retrieval).
- Docker (optional, only to run the container).
The first `uv sync` is slow because Sentence Transformers pulls in PyTorch. The first ingestion run also downloads the `BAAI/bge-small-en-v1.5` embedding model.
 
## Build
 
All build tasks are done with uv:
 
| Task | Command |
|---|---|
| Install dependencies and the project | `uv sync` |
| Update the lock file after editing `pyproject.toml` | `uv lock` |
| Add a dependency | `uv add <package>` |
| Show or change the project version | `uv version`, `uv version --bump patch` |
| Package the project (wheel + source archive) | `uv build` |
 
### Dependencies
 
Runtime dependencies are declared in `pyproject.toml` (`[project] dependencies`). The exact resolved versions are pinned in `uv.lock`, which is committed so that every machine installs the same set.
 
### Version
 
The version lives in one place, the `version` field of `pyproject.toml`. To change it:
 
```powershell
uv version                          # show the current version
uv version --dry-run --bump minor   # preview a change without writing it
uv version --bump patch             # 0.1.1 -> 0.1.2 (also: minor, major)
```
 
Then commit `pyproject.toml` and `uv.lock`, and tag the release (for example `git tag v0.1.2`).
 
### Packaging
 
```powershell
uv build
```
 
This creates two artifacts in `dist/`, both carrying the project version:
 
- `rag_assistant-<version>.tar.gz`: source distribution (includes `src`, `eval`, `README.md`, `pyproject.toml` and `uv.lock`).
- `rag_assistant-<version>-py3-none-any.whl`: wheel, which contains the `src` package.
The build backend is [Hatchling](https://hatch.pypa.io/), declared in `[build-system]`. Python code is not compiled, so building means producing these distributable artifacts. `dist/` is ignored by Git.
 
To check that the wheel installs in a clean environment (in PowerShell, replace `<version>` with the real version):
 
```powershell
uv venv test-env
uv pip install --python test-env dist\rag_assistant-<version>-py3-none-any.whl
test-env\Scripts\python -I -c "import src.config as c; print(c.__file__)"
```
 
The printed path must point inside `test-env\Lib\site-packages`, which shows the installed wheel was imported and not the local `src` folder.
 
## Corpus and indexing
 
To download up to 25 arXiv papers matching the default query and index their pages:
 
```powershell
uv run python -m src.ingest --download --max-results 25
```
 
Ingestion saves PDFs to `data/pdfs/`, stores their titles and arXiv IDs in `data/papers.json`, and creates the persistent index in `chroma_db/`. Downloaded PDFs and the index are ignored by Git. To index your own PDFs, place them in `data/pdfs/` and run `uv run python -m src.ingest`. Filenames are used as source IDs when arXiv metadata is unavailable.
 
The default chunk size is 1,000 characters with 200 characters of overlap. Chunking options are also available in `ingest_papers()` for controlled comparisons.
 
## Run
 
Vector search is available through `retrieve(question, k=5)` in `src.retrieve`. To generate a cited answer, configure your API key without adding it to the repository, either in the shell:
 
```powershell
$env:MISTRAL_API_KEY = "..."
```
 
or in a local `.env` file (loaded with `python-dotenv`, and not committed).
 
The default generation model is `open-mistral-nemo` (configured as `LLM_MODEL` in `src/config.py`).
 
Start the API:
 
```powershell
uv run uvicorn src.api:app --reload
```
 
Wait for "Application startup complete" (the first start can take a minute while the libraries load), then try `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/ask?q=...`. The response contains the generated answer and metadata for its source excerpts. The model is instructed to cite claims using `[n]` and explicitly state when information is missing.
 
## Test
 
The project does not have an automated test suite yet. These checks are available:
 
```powershell
uv run python -c "import src.config; print('ok')"    # the package imports from the uv environment
uv run python -m eval.run_eval                        # retrieval evaluation (see below)
uv run python scripts/check_mistral_models.py         # which Mistral models the API key can use (needs MISTRAL_API_KEY, makes real API calls)
```
 
### Retrieval evaluation
 
Add at least 20 questions with verified answers and the expected arXiv ID to `eval/questions.jsonl` (one JSON object per line):
 
```json
{"question":"What is LIGO's primary scientific goal?","source":"0711.3041"}
```
 
`uv run python -m eval.run_eval` reports hit@k, its 95% Wilson confidence interval, and MRR for k = 1, 3, 5, and 10. The included questions are a starting point, not a sufficiently large evaluation set for drawing conclusions. Evaluate multiple chunk sizes and embedding models separately; record results and failures after manual review.
 
## Project layout
 
```
pyproject.toml      project metadata, dependencies, build configuration
uv.lock             pinned dependency versions (committed)
.python-version     Python version used by uv (3.12)
src/                the package: ingest, retrieve, answer, api, config
eval/               retrieval evaluation (questions and script)
scripts/            helper scripts
data/pdfs/          downloaded papers (ignored by Git)
chroma_db/          vector index (ignored by Git)
Dockerfile          container image for the API
requirements.txt    export of uv.lock, used only by the Dockerfile
```
 
## Docker
 
Build the index locally first (see "Corpus and indexing"), then build and run the image from the project root:
 
```powershell
docker build -t rag-app .
docker run --rm -e MISTRAL_API_KEY -p 8000:8000 rag-app
```
 
The Dockerfile installs from `requirements.txt`, which is generated from `uv.lock` so the two stay in sync. After changing dependencies, regenerate it:
 
```powershell
uv export --no-hashes --no-emit-project -o requirements.txt
```
 
Never put the Mistral API key in the image or repository.
 