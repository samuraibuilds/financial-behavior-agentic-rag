
import pytest
from langchain_core.documents import Document

from src.agent import answer_question, _is_behavior_metrics_question


# 25 representative financial questions.
# These cases test routing intent; they do not claim to evaluate
# the financial accuracy of an external language model.
EVALUATION_CASES = [
    ("What was my spending?", True),
    ("What is my savings rate?", True),
    ("Show monthly income", True),
    ("What is the portfolio market value?", True),
    ("What is the allocation drift?", True),
    ("Explain portfolio concentration", True),
    ("What does HHI mean in this report?", True),
    ("What is my discretionary share?", True),
    ("Show cash flow", True),
    ("Compare actual and target weights", True),
    ("What is the savings change?", True),
    ("What is the latest income?", True),
    ("How much did I spend?", True),
    ("What is my asset allocation?", True),
    ("What is the portfolio value?", True),
    ("How does diversification reduce investment risk?", False),
    ("Explain liquidity risk", False),
    ("What is rebalancing?", False),
    ("Why can equity prices decline?", False),
    ("Does diversification eliminate market risk?", False),
    ("What is a bid-ask spread?", False),
    ("Explain the role of interest rates", False),
    ("What is an investment time horizon?", False),
    ("Which stock will rise tomorrow?", False),
    ("What will the market return next year?", False),
]


@pytest.mark.parametrize(
    "question, expected",
    EVALUATION_CASES,
    ids=[case[0] for case in EVALUATION_CASES],
)
def test_question_routing_benchmark(question, expected):
    assert _is_behavior_metrics_question(question) is expected


def test_no_retrieved_documents_causes_abstention(monkeypatch):
    monkeypatch.setattr("src.agent.load_vector_store", lambda: object())
    monkeypatch.setattr("src.agent.search_documents", lambda *args, **kwargs: [])

    def llm_must_not_run():
        raise AssertionError("LLM must not answer without retrieved evidence")

    monkeypatch.setattr("src.agent.get_llm", llm_must_not_run)

    result = answer_question(
        "What does the document say about an unknown investment product?",
        include_behavior=False,
    )

    assert "can't answer" in result["answer"].lower()
    assert result["sources"] == []
    assert result["retrieved_documents"] == []


def test_diversification_answer_uses_retrieved_evidence(monkeypatch):
    document = Document(
        page_content=(
            "Diversification spreads investments across different assets, "
            "sectors, and markets. It can reduce concentration risk, but "
            "it does not eliminate market risk."
        ),
        metadata={"source": "sample_financial_notes.txt"},
    )

    monkeypatch.setattr("src.agent.load_vector_store", lambda: object())
    monkeypatch.setattr(
        "src.agent.search_documents",
        lambda *args, **kwargs: [document],
    )

    result = answer_question(
        "How does diversification reduce investment risk?",
        top_k=1,
    )

    assert "concentration risk" in result["answer"].lower()
    assert "does not eliminate market risk" in result["answer"].lower()
    assert "sample_financial_notes.txt" in result["sources"]


def test_unsupported_question_does_not_claim_document_sources(monkeypatch):
    monkeypatch.setattr("src.agent.load_vector_store", lambda: object())
    monkeypatch.setattr("src.agent.search_documents", lambda *args, **kwargs: [])

    result = answer_question(
        "Predict the exact return of an unknown asset next year",
        include_behavior=False,
    )

    assert "can't answer" in result["answer"].lower()
    assert result["sources"] == []


def test_metric_question_is_routed_to_calculated_metrics():
    assert _is_behavior_metrics_question(
        "What is the latest savings rate?"
    )
    assert _is_behavior_metrics_question(
        "Which asset has the largest allocation drift?"
    )


def test_unrelated_question_is_not_routed_to_behavior_metrics():
    assert not _is_behavior_metrics_question(
        "How does diversification reduce investment risk?"
    )
