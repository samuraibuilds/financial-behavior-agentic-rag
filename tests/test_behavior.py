import pytest

from src.behavior import (
    analyze_cashflow,
    analyze_portfolio,
    build_behavior_report,
    load_behavior_data,
)


def test_synthetic_behavior_report_has_expected_structure():
    report = build_behavior_report()

    assert report["data_status"] == "synthetic demonstration data"
    assert report["cashflow"]["month"] == "2026-10"
    assert report["portfolio"]["month"] == "2026-10"
    assert report["portfolio"]["total_market_value"] == 88000


def test_savings_rate_is_calculated_from_data():
    cashflow, _ = load_behavior_data()
    result = analyze_cashflow(cashflow)

    assert result["savings_rate_pct"] == pytest.approx(round(100 * 1700 / 5400, 2))
    assert result["total_spending"] == 3700


def test_portfolio_weights_sum_to_100_percent():
    _, portfolio = load_behavior_data()
    result = analyze_portfolio(portfolio)

    actual_total = sum(
        item["actual_weight_pct"] for item in result["allocations"]
    )
    assert actual_total == pytest.approx(100, abs=0.02)
    assert result["one_way_allocation_drift_pct"] >= 0


def test_missing_dataset_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing"):
        load_behavior_data(tmp_path)
