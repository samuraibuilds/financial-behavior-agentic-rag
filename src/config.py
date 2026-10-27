import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
)
OLLAMA_CHAT_MODEL = os.getenv(
    "OLLAMA_CHAT_MODEL", "llama3.2:1b"
)
OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
)

MARKET_TICKERS = [
    ticker.strip()
    for ticker in os.getenv(
        "MARKET_TICKERS", "SPY,QQQ,AAPL,MSFT"
    ).split(",")
    if ticker.strip()
]

MARKET_PERIOD = os.getenv("MARKET_PERIOD", "6mo")
MARKET_INTERVAL = os.getenv("MARKET_INTERVAL", "1d")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be positive")

if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be between zero and CHUNK_SIZE"
    )

if RETRIEVAL_TOP_K <= 0:
    raise ValueError("RETRIEVAL_TOP_K must be positive")


def get_llm():
    return ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def get_embeddings():
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )