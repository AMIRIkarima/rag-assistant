import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMBEDDING_MODEL,
    PDF_DIR,
    PAPER_METADATA_PATH,
)


def chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    if size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if overlap < 0 or overlap >= size:
        raise ValueError("Chunk overlap must be non-negative and smaller than the chunk size.")
    step = size - overlap
    return [text[start : start + size] for start in range(0, len(text), step) if text[start : start + size].strip()]


def _arxiv_source_id(value: str) -> str:
    return re.sub(r"v\d+$", "", value.removeprefix("https://arxiv.org/abs/"))


def _download_pdf(url: str, output_path: Path) -> None:
    request = Request(url, headers={"User-Agent": "scientific-paper-rag/1.0"})
    with urlopen(request, timeout=60) as response:
        output_path.write_bytes(response.read())


def download_papers(max_results: int = 25, query: str = "gravitational waves LIGO detection") -> int:
    if max_results <= 0:
        raise ValueError("max_results must be greater than zero.")

    import arxiv

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict[str, str]] = {}
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    for paper in client.results(search):
        short_id = paper.get_short_id()
        if not paper.pdf_url:
            raise ValueError(f"arXiv result {short_id} has no PDF URL.")
        filename = f"{re.sub(r'[\\/]', '_', short_id)}.pdf"
        _download_pdf(paper.pdf_url, PDF_DIR / filename)
        metadata[filename] = {
            "source": _arxiv_source_id(short_id),
            "title": paper.title,
        }

    PAPER_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAPER_METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(metadata)


def _load_metadata() -> dict[str, dict[str, str]]:
    if not PAPER_METADATA_PATH.exists():
        return {}
    with PAPER_METADATA_PATH.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object in {PAPER_METADATA_PATH}.")
    return metadata


def ingest_papers(
    pdf_dir: Path = PDF_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> int:
    import chromadb
    import fitz
    from sentence_transformers import SentenceTransformer

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}. Add PDFs or run with --download.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    paper_metadata = _load_metadata()
    indexed = 0

    for pdf_path in pdfs:
        file_metadata: dict[str, Any] = paper_metadata.get(pdf_path.name, {})
        source = str(file_metadata.get("source", _arxiv_source_id(pdf_path.stem)))
        title = str(file_metadata.get("title", pdf_path.stem))
        with fitz.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                chunks = [
                    chunk
                    for chunk in chunk_text(page.get_text(), chunk_size, chunk_overlap)
                    if len(chunk.strip()) >= 200
                ]
                if not chunks:
                    continue

                ids = [f"{pdf_path.stem}_p{page_number}_{index}" for index in range(len(chunks))]
                metadatas = [
                    {"source": source, "title": title, "page": page_number}
                    for _ in chunks
                ]
                embeddings = model.encode(chunks).tolist()
                collection.upsert(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                indexed += len(chunks)

    if indexed == 0:
        raise ValueError(f"No text chunks of at least 200 characters were extracted from PDFs in {pdf_dir}.")
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download arXiv papers and index local PDFs.")
    parser.add_argument("--download", action="store_true", help="Download relevant papers from arXiv first.")
    parser.add_argument("--max-results", type=int, default=25)
    parser.add_argument("--query", default="gravitational waves LIGO detection")
    args = parser.parse_args()

    if args.download:
        print(f"Downloaded {download_papers(args.max_results, args.query)} papers.")
    print(f"Indexed {ingest_papers()} text chunks.")


if __name__ == "__main__":
    main()