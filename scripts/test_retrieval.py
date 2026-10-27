
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import load_vector_store


def main():
    store = load_vector_store()
    question = "How does diversification reduce investment risk?"

    results = store.similarity_search_with_score(question, k=4)

    print(f"Question: {question}")
    print(f"Results returned: {len(results)}")

    for rank, (document, distance) in enumerate(results, start=1):
        print(f"\n--- RESULT {rank} | DISTANCE {distance:.4f} ---")
        print(document.page_content)
        print("SOURCE:", document.metadata.get("source"))


if __name__ == "__main__":
    main()
