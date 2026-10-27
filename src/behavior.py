from pathlib import Path

import pandas as pd

from src.config import BASE_DIR

DATA_DIR = BASE_DIR / "data"

CASHFLOW_COLUMNS = {
    "month", "income", "essential_spending",
    "discretionary_spending", "savings",
}
PORTFOLIO_COLUMNS = {
    "month", "asset_class", "market_value", "target_weight",
}


def load_behavior_data(data_dir: Path = DATA_DIR):
    """Load and validate the synthetic financial activity datasets."""
    cashflow_path = data_dir / "monthly_cashflow.csv"
    portfolio_path = data_dir / "portfolio_snapshots.csv"

    if not cashflow_path.exists() or not portfolio_path.exists():
        raise FileNotFoundError(
            "Missing monthly_cashflow.csv or portfolio_snapshots.csv "
            f"in {data_dir}"
        )

    cashflow = pd.read_csv(cashflow_path)
    portfolio = pd.read_csv(portfolio_path)

    if not CASHFLOW_COLUMNS.issubset(cashflow.columns):
        raise ValueError("Cash-flow data is missing required columns.")
    if not PORTFOLIO_COLUMNS.issubset(portfolio.columns):
        raise ValueError("Portfolio data is missing required columns.")
    if cashflow.empty or portfolio.empty:
        raise ValueError("Financial activity datasets cannot be empty.")

    for column in (
        "income", "essential_spending",
        "discretionary_spending", "savings",
    ):
        cashflow[column] = pd.to_numeric(
            cashflow[column], errors="raise"
        )

    portfolio["market_value"] = pd.to_numeric(
        portfolio["market_value"], errors="raise"
    )
    portfolio["target_weight"] = pd.to_numeric(
        portfolio["target_weight"], errors="raise"
    )

    if cashflow[list(CASHFLOW_COLUMNS - {"month"})].isna().any().any():
        raise ValueError("Cash-flow data contains missing numeric values.")
    if portfolio[
        ["market_value", "target_weight"]
    ].isna().any().any():
        raise ValueError("Portfolio data contains missing numeric values.")

    if (cashflow["income"] <= 0).any():
        raise ValueError("Monthly income must be greater than zero.")
    if (cashflow[
        ["essential_spending", "discretionary_spending", "savings"]
    ] < 0).any().any():
        raise ValueError("Spending and savings cannot be negative.")
    if (portfolio["market_value"] < 0).any():
        raise ValueError("Market values cannot be negative.")
    if not portfolio["target_weight"].between(0, 1).all():
        raise ValueError("Target weights must be between zero and one.")

    cashflow["month"] = cashflow["month"].astype(str)
    portfolio["month"] = portfolio["month"].astype(str)

    if cashflow["month"].duplicated().any():
        raise ValueError("Cash-flow data must contain one row per month.")
    if portfolio.duplicated(["month", "asset_class"]).any():
        raise ValueError("Duplicate asset class within a portfolio month.")

    for month, group in portfolio.groupby("month"):
        if group["market_value"].sum() <= 0:
            raise ValueError(f"Portfolio value must be positive in {month}.")
        if abs(group["target_weight"].sum() - 1.0) > 0.001:
            raise ValueError(
                f"Target weights for {month} must sum to 1.0."
            )

    return (
        cashflow.sort_values("month").reset_index(drop=True),
        portfolio.sort_values(["month", "asset_class"]).reset_index(drop=True),
    )


def analyze_cashflow(cashflow: pd.DataFrame) -> dict:
    """Calculate cash-flow indicators; no language model is used."""
    current = cashflow.iloc[-1]
    previous = cashflow.iloc[-2] if len(cashflow) > 1 else None

    income = float(current["income"])
    essential = float(current["essential_spending"])
    discretionary = float(current["discretionary_spending"])
    savings = float(current["savings"])
    spending = essential + discretionary

    result = {
        "month": str(current["month"]),
        "income": income,
        "essential_spending": essential,
        "discretionary_spending": discretionary,
        "total_spending": spending,
        "savings": savings,
        "savings_rate_pct": round(100 * savings / income, 2),
        "discretionary_share_pct": round(
            100 * discretionary / spending, 2
        ) if spending else 0.0,
        "unallocated_cashflow": round(income - spending - savings, 2),
        "spending_change_pct": None,
        "savings_rate_change_pp": None,
    }

    if previous is not None:
        previous_spending = float(
            previous["essential_spending"]
            + previous["discretionary_spending"]
        )
        previous_income = float(previous["income"])
        previous_savings_rate = (
            100 * float(previous["savings"]) / previous_income
        )

        result["spending_change_pct"] = (
            round(100 * (spending - previous_spending) / previous_spending, 2)
            if previous_spending else None
        )
        result["savings_rate_change_pp"] = round(
            result["savings_rate_pct"] - previous_savings_rate, 2
        )

    return result


def analyze_portfolio(portfolio: pd.DataFrame) -> dict:
    """Measure latest allocation, target drift and concentration."""
    months = sorted(portfolio["month"].unique())
    current_month = months[-1]
    current = portfolio[portfolio["month"] == current_month].copy()

    total_value = float(current["market_value"].sum())
    current["actual_weight"] = current["market_value"] / total_value
    current["drift_pp"] = (
        100 * (current["actual_weight"] - current["target_weight"])
    )

    hhi = float((current["actual_weight"] ** 2).sum())
    assets = current.sort_values(
        "actual_weight", ascending=False
    )

    allocations = []
    for _, row in assets.iterrows():
        allocations.append({
            "asset_class": str(row["asset_class"]),
            "market_value": round(float(row["market_value"]), 2),
            "actual_weight_pct": round(
                100 * float(row["actual_weight"]), 2
            ),
            "target_weight_pct": round(
                100 * float(row["target_weight"]), 2
            ),
            "drift_pp": round(float(row["drift_pp"]), 2),
            "change_from_previous_month_pp": None,
        })

    if len(months) > 1:
        previous = portfolio[portfolio["month"] == months[-2]].copy()
        previous_total = float(previous["market_value"].sum())
        previous["actual_weight"] = (
            previous["market_value"] / previous_total
        )
        previous_weights = dict(zip(
            previous["asset_class"], previous["actual_weight"]
        ))

        for item in allocations:
            old_weight = previous_weights.get(item["asset_class"])
            if old_weight is not None:
                item["change_from_previous_month_pp"] = round(
                    item["actual_weight_pct"] - 100 * float(old_weight), 2
                )

    return {
        "month": current_month,
        "total_market_value": round(total_value, 2),
        "concentration_hhi": round(hhi, 4),
        "effective_number_of_asset_classes": round(1 / hhi, 2) if hhi else 0,
        "one_way_allocation_drift_pct": round(
            0.5 * current["drift_pp"].abs().sum(), 2
        ),
        "allocations": allocations,
        "interpretation_note": (
            "HHI and allocation drift are descriptive indicators, not "
            "complete measures of portfolio risk. Asset-class groupings "
            "do not reveal all underlying holdings or correlations."
        ),
    }


def build_behavior_report(data_dir: Path = DATA_DIR) -> dict:
    """Return calculated indicators with their synthetic-data provenance."""
    cashflow, portfolio = load_behavior_data(data_dir)

    return {
        "data_status": "synthetic demonstration data",
        "cashflow": analyze_cashflow(cashflow),
        "portfolio": analyze_portfolio(portfolio),
    }
