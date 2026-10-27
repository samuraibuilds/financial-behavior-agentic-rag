
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import RETRIEVAL_TOP_K, get_embeddings

BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = BASE_DIR / "storage" / "faiss_index"


def build_vector_store(chunks: list[Document]) -> FAISS:
    """Create a FAISS vector store from document chunks."""
    if not chunks:
        raise ValueError("Cannot build a vector store from empty chunks.")

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


def save_vector_store(
    vector_store: FAISS,
    index_dir: Path = INDEX_DIR,
) -> None:
    """Persist the FAISS index locally."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(index_dir))


def load_vector_store(
    index_dir: Path = INDEX_DIR,
) -> FAISS:
    """Load a trusted FAISS index created by this project."""
    index_dir = Path(index_dir)

    if not (index_dir / "index.faiss").is_file():
        raise FileNotFoundError(
            f"FAISS index not found in {index_dir}. "
            "Build and save the index first."
        )

    if not (index_dir / "index.pkl").is_file():
        raise FileNotFoundError(
            f"FAISS metadata file not found in {index_dir}."
        )

    return FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )

def search_documents(
    vector_store: FAISS,
    query: str,
    k: int = RETRIEVAL_TOP_K,
) -> list[Document]:
    """Retrieve the most relevant document chunks."""
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k <= 0:
        raise ValueError("k must be positive.")

    return vector_store.similarity_search(query, k=k)
