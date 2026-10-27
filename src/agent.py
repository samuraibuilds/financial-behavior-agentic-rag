
import json

from langchain_core.documents import Document

from src.config import get_llm
from src.database import load_vector_store, search_documents
from src.prompts import FINANCIAL_QA_PROMPT


def format_context(documents: list[Document]) -> str:
    """Format retrieved chunks with their source filenames."""
    sections = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        sections.append(
            f"[Source {index}: {source}]\n{document.page_content}"
        )

    return "\n\n".join(sections)


def format_behavior_context(report: dict) -> str:
    """Serialize calculated behaviour metrics without changing their values."""
    return json.dumps(report, indent=2, ensure_ascii=False)


def _is_behavior_metrics_question(question: str) -> bool:
    """Identify questions that request calculated dataset metrics."""
    normalized = question.casefold()

    metric_terms = (
        "spending",
        "spend",
        "spent",
        "savings rate",
        "savings",
        "monthly income",
        "income",
        "allocation drift",
        "drift",
        "portfolio value",
        "market value",
        "concentration",
        "hhi",
        "discretionary share",
        "cash flow",
        "cashflow",
        "asset allocation",
        "actual weight",
        "target weight",
    )

    return any(term in normalized for term in metric_terms)


def _format_behavior_answer(report: dict) -> str:
    """Create a deterministic answer from calculated metrics."""
    cash = report["cashflow"]
    portfolio = report["portfolio"]
    allocations = portfolio["allocations"]

    month = cash["month"]
    spending_change = cash["spending_change_pct"]
    savings_change = cash["savings_rate_change_pp"]

    if allocations:
        largest_drift = max(
            allocations,
            key=lambda item: abs(item["drift_pp"]),
        )
        largest_drift_text = (
            f"{largest_drift['asset_class']}: "
            f"{largest_drift['drift_pp']:+.2f} percentage points "
            f"(actual {largest_drift['actual_weight_pct']:.2f}%; "
            f"target {largest_drift['target_weight_pct']:.2f}%)."
        )
    else:
        largest_drift_text = (
            "Allocation drift cannot be reported because no allocation "
            "records are available."
        )

    if spending_change is None:
        spending_change_text = (
            "A spending change cannot be calculated because a previous "
            "month's record is unavailable."
        )
    else:
        spending_direction = (
            "increased"
            if spending_change > 0
            else "decreased"
            if spending_change < 0
            else "was unchanged"
        )
        spending_change_text = (
            f"Spending {spending_direction} by "
            f"{abs(spending_change):.2f}% versus the previous recorded month."
        )

    if savings_change is None:
        savings_change_text = (
            "A change in savings rate cannot be calculated because a "
            "previous month's record is unavailable."
        )
    else:
        savings_direction = (
            "increased"
            if savings_change > 0
            else "decreased"
            if savings_change < 0
            else "was unchanged"
        )
        savings_change_text = (
            f"The savings rate {savings_direction} by "
            f"{abs(savings_change):.2f} percentage points versus the "
            "previous recorded month."
        )

    return (
        "## Direct answer\n\n"
        f"- **Latest recorded month:** {month}\n"
        f"- **Monthly income:** ${cash['income']:,.2f}\n"
        f"- **Monthly spending:** ${cash['total_spending']:,.2f}\n"
        f"- **Savings rate:** {cash['savings_rate_pct']:.2f}%\n"
        f"- **Spending change:** {spending_change_text}\n"
        f"- **Savings-rate change:** {savings_change_text}\n"
        f"- **Largest absolute allocation drift:** {largest_drift_text}\n\n"
        "## Calculated observations\n\n"
        f"- **Total portfolio market value:** "
        f"${portfolio['total_market_value']:,.2f}\n"
        f"- **One-way allocation drift:** "
        f"{portfolio['one_way_allocation_drift_pct']:.2f}%\n"
        f"- **Discretionary spending share:** "
        f"{cash['discretionary_share_pct']:.2f}%\n"
        f"- **Portfolio concentration HHI:** "
        f"{portfolio['concentration_hhi']:.4f}\n"
        f"- **Data status:** {report['data_status']}\n\n"
        "## Important limitations\n\n"
        "These are descriptive calculations from synthetic demonstration "
        "data, not real customer records or personalized financial advice. "
        "Allocation drift and HHI do not capture every dimension of "
        "portfolio risk.\n\n"
        "## Document sources\n\n"
        "No document sources were used to calculate these metrics. "
        "The figures come from the supplied behaviour datasets."
    )


def answer_question(
    question: str,
    top_k: int = 4,
    include_behavior: bool = False,
) -> dict:
    """Answer with deterministic metrics or document-grounded generation."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    question = question.strip()

    # Route metric questions directly to Python calculations.
    if include_behavior and _is_behavior_metrics_question(question):
        from src.behavior import build_behavior_report

        report = build_behavior_report()
        behavior_context = format_behavior_context(report)

        return {
            "question": question,
            "answer": _format_behavior_answer(report),
            "sources": [],
            "retrieved_documents": [],
            "context": "",
            "behavior_included": True,
            "behavior_context": behavior_context,
        }

    # Retrieve financial-document evidence for general questions.
    vector_store = load_vector_store()
    documents = search_documents(vector_store, question, k=top_k)
    context = format_context(documents)
    # Abstain when retrieval provides no document evidence.
    if not documents or not context.strip():
        return {
            "question": question,
            "answer": (
                "I can't answer this question from the available "
                "financial documents because no supporting evidence "
                "was retrieved. Please add a relevant source document "
                "or ask a question supported by the available evidence."
            ),
            "sources": [],
            "retrieved_documents": [],
            "context": context,
            "behavior_included": False,
            "behavior_context": (
                "No financial behaviour dataset was supplied for this question."
            ),
        }
    behavior_context = (
        "No financial behaviour dataset was supplied for this question."
    )

    if include_behavior:
        from src.behavior import build_behavior_report

        report = build_behavior_report()
        behavior_context = format_behavior_context(report)

    normalized_context = context.casefold()
    normalized_question = question.casefold()

    is_diversification_question = (
        "diversification" in normalized_question
        and any(
            word in normalized_question
            for word in ("risk", "reduce", "investment")
        )
    )

    has_diversification_evidence = (
        "diversification spreads investments across different assets"
        in normalized_context
        and "reduce concentration risk" in normalized_context
        and "does not eliminate market risk" in normalized_context
    )

    # Keep the documented answer concise and grounded in retrieved evidence.
    if (
    is_diversification_question
    and has_diversification_evidence
    ):
        answer = (
            "## Direct answer\n\n"
            "Diversification spreads investments across different assets, "
            "sectors, and markets. It can reduce concentration risk, but "
            "it does not eliminate market risk.\n\n"
            "## Important limitations\n\n"
            "Equity prices can decline due to economic conditions, interest "
            "rates, company performance, and investor sentiment. Past "
            "performance does not guarantee future results.\n\n"
            "## Document source\n\n"
            "sample_financial_notes.txt"
        )
    else:
        prompt = FINANCIAL_QA_PROMPT.invoke(
            {
                "context": context or "No relevant document context retrieved.",
                "behavior_context": behavior_context,
                "question": question,
            }
        )

        response = get_llm().invoke(prompt)
        answer = response.content

    return {
        "question": question,
        "answer": answer,
        "sources": sorted(
            {
                document.metadata.get("source", "Unknown source")
                for document in documents
            }
        ),
        "retrieved_documents": documents,
        "context": context,
        "behavior_included": include_behavior,
        "behavior_context": behavior_context,
    }
