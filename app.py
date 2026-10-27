
import streamlit as st
import pandas as pd
import plotly.express as px

from src.agent import answer_question
from src.behavior import build_behavior_report


st.set_page_config(
    page_title="Financial Behaviour Intelligence RAG",
    page_icon="📊",
    layout="wide",
)

st.title("Financial Behaviour Intelligence RAG")
st.caption(
    "Financial activity analytics and evidence-grounded "
    "answers using a local retrieval-augmented generation system."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved document chunks", 1, 8, 4)
    st.warning(
        "Synthetic demonstration data only. "
        "Not personalized financial advice."
    )

dashboard_tab, assistant_tab, methodology_tab = st.tabs(
    ["📊 Behaviour Dashboard", "💬 Financial RAG", "ℹ️ Methodology"]
)

with dashboard_tab:
    st.subheader("Financial behaviour overview")

    try:
        report = build_behavior_report()
        cash = report["cashflow"]
        portfolio = report["portfolio"]

        st.info(
            "All displayed financial activity is synthetic demonstration "
            "data. These metrics do not describe a real person or account."
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Monthly income",
            f"${cash['income']:,.0f}",
        )

        col2.metric(
            "Savings rate",
            f"{cash['savings_rate_pct']:.2f}%",
            (
                f"{cash['savings_rate_change_pp']:+.2f} percentage points"
                if cash["savings_rate_change_pp"] is not None
                else None
            ),
        )

        col3.metric(
            "Monthly spending",
            f"${cash['total_spending']:,.0f}",
            (
                f"{cash['spending_change_pct']:+.2f}% vs prior month"
                if cash["spending_change_pct"] is not None
                else None
            ),
            delta_color="inverse",
        )

        col4.metric(
            "Portfolio value",
            f"${portfolio['total_market_value']:,.0f}",
        )

        st.divider()

        left, right = st.columns(2)

        with left:
            st.subheader("Monthly cash flow")

            cashflow_df = pd.read_csv("data/monthly_cashflow.csv")

            cashflow_df["total_spending"] = (
                cashflow_df["essential_spending"]
                + cashflow_df["discretionary_spending"]
            )

            chart_df = cashflow_df.melt(
                id_vars="month",
                value_vars=[
                    "income",
                    "total_spending",
                    "savings",
                ],
                var_name="category",
                value_name="amount",
            )

            fig = px.line(
                chart_df,
                x="month",
                y="amount",
                color="category",
                markers=True,
                title="Income, spending and savings",
                labels={
                    "month": "Month",
                    "amount": "Amount ($)",
                    "category": "Measure",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Latest portfolio allocation")

            allocations_df = pd.DataFrame(portfolio["allocations"])

            fig = px.bar(
                allocations_df,
                x="asset_class",
                y=["actual_weight_pct", "target_weight_pct"],
                barmode="group",
                title="Actual vs target allocation",
                labels={
                    "asset_class": "Asset class",
                    "value": "Allocation (%)",
                    "variable": "Measure",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Allocation and drift details")

        display_df = allocations_df.rename(
            columns={
                "asset_class": "Asset class",
                "market_value": "Market value ($)",
                "actual_weight_pct": "Actual weight (%)",
                "target_weight_pct": "Target weight (%)",
                "drift_pp": "Drift (percentage points)",
                "change_from_previous_month_pp":
                    "Monthly change (percentage points)",
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Calculated indicators")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Discretionary spending share",
            f"{cash['discretionary_share_pct']:.2f}%",
        )

        col2.metric(
            "One-way allocation drift",
            f"{portfolio['one_way_allocation_drift_pct']:.2f}%",
        )

        col3.metric(
            "Concentration HHI",
            f"{portfolio['concentration_hhi']:.4f}",
        )

        st.caption(
            "HHI is the sum of squared asset-class weights. Allocation drift "
            "is half the sum of absolute differences between actual and "
            "target weights. Neither metric alone measures total portfolio risk."
        )

        st.subheader("Evidence-based observations")

        if cash["spending_change_pct"] is not None:
            direction = (
                "increased"
                if cash["spending_change_pct"] > 0
                else "decreased"
                if cash["spending_change_pct"] < 0
                else "was unchanged"
            )

            st.write(
                f"- Total spending {direction} by "
                f"{abs(cash['spending_change_pct']):.2f}% versus the prior "
                "recorded month."
            )

        st.write(
            f"- The latest recorded savings rate is "
            f"{cash['savings_rate_pct']:.2f}% of income."
        )

        if portfolio["allocations"]:
            largest_drift = max(
                portfolio["allocations"],
                key=lambda item: abs(item["drift_pp"]),
            )

            st.write(
                f"- The largest absolute allocation drift is in "
                f"{largest_drift['asset_class']}: "
                f"{largest_drift['drift_pp']:+.2f} percentage points "
                "versus target."
            )

        st.caption(
            "These are descriptions of the supplied synthetic records, "
            "not predictions or personalized recommendations."
        )

    except Exception as exc:
        st.error(f"Could not load the behaviour dashboard: {exc}")


with assistant_tab:
    st.subheader("Ask your financial documents")

    question = st.text_area(
        "Your financial question",
        placeholder=(
            "Example: What happened to spending in the latest month, "
            "what is the savings rate, and which asset class has the "
            "largest allocation drift?"
        ),
        key="financial_question",
    )

    include_behavior = st.checkbox(
        "Include calculated financial behaviour metrics",
        value=True,
        help=(
            "Allows the assistant to use calculated metrics from the "
            "synthetic cashflow and portfolio datasets."
        ),
    )

    if st.button("Ask question", type="primary"):
        if not question.strip():
            st.error("Please enter a question.")

        else:
            try:
                with st.spinner(
                    "Retrieving evidence and preparing an answer..."
                ):
                    result = answer_question(
                        question,
                        top_k=top_k,
                        include_behavior=include_behavior,
                    )

                st.markdown("#### Answer")
                st.markdown(result["answer"])



                sources = result.get("sources", [])
                behavior_used = result.get("behavior_included", False)

                if sources:
                    st.markdown("#### Retrieved document sources")
                    for source in sources:
                        st.write(f"- `{source}`")
                elif behavior_used:
                    st.caption(
                        "No document sources were used for the calculated "
                        "behaviour metrics."
                    )
                else:
                    st.markdown("#### Retrieved document sources")
                    st.write("No source documents were retrieved.")


                if behavior_used:
                    st.caption(
                        "Calculated behaviour metrics are based on "
                        "synthetic demonstration data."
                    )

                    with st.expander(
                        "Inspect calculated behaviour context"
                    ):
                        st.code(
                            result.get(
                                "behavior_context",
                                "Behaviour context was not returned.",
                            ),
                            language="json",
                        )

                with st.expander("Inspect retrieved document context"):
                    st.text(
                        result.get("context", "")
                        or "No document context retrieved."
                    )

            except Exception as exc:
                st.error(f"Could not answer the question: {exc}")


with methodology_tab:
    st.subheader("How the system works")

    st.markdown(
        """
        1. **Structured analytics:** Pandas calculates financial indicators
           directly from the synthetic CSV datasets.
        2. **Document retrieval:** FAISS retrieves relevant chunks from
           the indexed financial notes.
        3. **Language-model response:** Ollama generates answers using the
           retrieved document context and, when enabled, calculated
           behaviour metrics.
        4. **Traceability:** The dashboard shows calculated inputs, while
           the document assistant displays retrieved source filenames
           and context.
        5. **Optional behaviour context:** The Financial RAG tab can pass
           the calculated cashflow and portfolio report to the language
           model for questions involving the supplied datasets.

        **Current limitations**

        - The behaviour datasets are synthetic, not connected to real accounts.
        - The dashboard calculates descriptive indicators, not future returns.
        - Answers depend on the retrieved document evidence and supplied
          calculated metrics. The language model can still make mistakes.
        - Portfolio concentration at asset-class level does not capture
          individual security concentration, correlations, or all risk factors.
        - Outputs are educational demonstrations, not investment advice.
        """
    )
