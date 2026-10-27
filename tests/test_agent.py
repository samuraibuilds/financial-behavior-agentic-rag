from src.agent import answer_question


def test_diversification_answer_preserves_risk_limitation():
    result = answer_question(
        "How does diversification reduce investment risk?",
        top_k=1,
    )

    answer = result["answer"].lower()

    assert "concentration risk" in answer
    assert "does not eliminate market risk" in answer
    assert "sample_financial_notes.txt" in answer


def test_empty_question_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer_question("   ")
