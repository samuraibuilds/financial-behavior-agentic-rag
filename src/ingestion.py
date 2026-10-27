
from pathlib import Path

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load supported financial documents recursively from a directory."""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {data_dir}"
        )

    documents: list[Document] = []

    for file_path in sorted(data_dir.rglob("*")):
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue

        if file_path.stat().st_size == 0:
            print(f"Skipping empty file: {file_path.name}")
            continue

        try:
            if extension == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif extension == ".csv":
                loader = CSVLoader(
                    str(file_path),
                    encoding="utf-8",
                )
            else:
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8",
                    autodetect_encoding=True,
                )

            loaded = loader.load()

            for document in loaded:
                document.metadata["source"] = str(
                    file_path.relative_to(data_dir)
                )

            documents.extend(loaded)
            print(f"Loaded {len(loaded)} document page(s)/record(s): {file_path.name}")

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {file_path}: {exc}"
            ) from exc

    if not documents:
        raise ValueError(
            f"No readable PDF, TXT, MD, or CSV files found in {data_dir}. "
            "Add at least one non-empty supported document."
        )

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """Split loaded documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    chunks = [
        chunk for chunk in chunks if chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError(
            "Document splitting produced no non-empty chunks."
        )

    return chunks
