from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
PAPER_METADATA_PATH = PROJECT_ROOT / "data" / "papers.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
