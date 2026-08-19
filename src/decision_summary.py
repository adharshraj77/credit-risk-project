"""
Phase 8: Decision Summary Layer
=================================
Generates templated natural language decision summaries from SHAP values,
counterfactual recommendations, and scenario simulation results.
Pure template-based — no LLM, no hallucination risk.

Usage:
    from src.decision_summary import generate_summary
    summary = generate_summary(prob, shap_drivers, counterfactuals, scenario_results)
"""

import numpy as np


def _risk_tier(probability):
    """Classify probability into risk tiers."""
    if probability < 0.15:
        return "Low Risk", "🟢"
    elif probability < 0.35:
        return "Medium Risk", "🟡"
    elif probability < 0.55:
        return "High Risk", "🟠"
    else:
        return "Very High Risk", "🔴"


def _format_number(value, is_currency=False, is_pct=False):
    """Format a number for display."""
    if value is None:
        return "N/A"
    if is_currency:
        return f"${value:,.0f}"
    if is_pct:
        return f"{value:.1f}%"
    return f"{value:,.2f}"


def _format_feature_name(feature):
    """Convert feature name to human-readable text."""
    name_map = {
        "loan_amnt": "loan amount",
        "annual_inc": "annual income",
        "int_rate": "interest rate",
        "dti": "debt-to-income ratio",
        "revol_util": "revolving utilization",
        "revol_bal": "revolving balance",
        "installment": "monthly installment",
        "open_acc": "open accounts",
        "total_acc": "total accounts",
        "credit_history_months": "credit history length",
        "loan_to_income_ratio": "loan-to-income ratio",
        "payment_to_income_ratio": "payment-to-income ratio",
        "credit_utilization_ratio": "credit utilization ratio",
        "delinquency_ratio": "delinquency ratio",
        "emp_length": "employment length",
        "pub_rec": "public records",
        "delinq_2yrs": "recent delinquencies",
        "inq_last_6mths": "recent credit inquiries",
        "int_rate_x_dti": "rate × DTI interaction",
        "rate_burden": "rate burden score",
        "sub_grade_encoded": "credit sub-grade",
        "addr_state_encoded": "geographic risk factor",
        "purpose_encoded": "loan purpose risk factor",
        "mths_since_last_delinq": "months since last delinquency",
        "mths_since_last_delinq_was_missing": "no delinquency history",
        "revol_bal_to_income": "revolving balance to income",
        "open_acc_ratio": "open account ratio",
        "available_credit_pct": "available credit percentage",
    }
    return name_map.get(feature, feature.replace("_", " "))


def generate_risk_assessment(probability, top_drivers):
    """
    Generate a risk assessment summary from the model prediction and SHAP drivers.

    Parameters:
        probability: float, default probability (0–1)
        top_drivers: list of dicts with keys: feature, shap_value, direction, feature_value

    Returns:
        str: Multi-paragraph risk assessment text
    """
    tier, emoji = _risk_tier(probability)

    # Opening statement
    lines = [
        f"## {emoji} Risk Assessment: {tier}",
        "",
        f"This applicant has a **{probability:.1%}** probability of default, "
        f"placing them in the **{tier}** category.",
        "",
    ]

    if not top_drivers:
        lines.append("_SHAP analysis not available for this assessment._")
        return "\n".join(lines)

    # Risk drivers
    increasing = [d for d in top_drivers if d["shap_value"] > 0]
    decreasing = [d for d in top_drivers if d["shap_value"] <= 0]

    if increasing:
        lines.append("### Risk-Increasing Factors")
        lines.append("")
        for d in increasing:
            name = _format_feature_name(d["feature"])
            val = d.get("feature_value")
            val_str = f" (current value: {_format_number(val)})" if val is not None else ""
            contribution = abs(d["shap_value"])
            lines.append(
                f"- **{name.title()}**{val_str}: "
                f"increases default risk by {contribution:.3f} SHAP units"
            )
        lines.append("")

    if decreasing:
        lines.append("### Risk-Decreasing Factors")
        lines.append("")
        for d in decreasing:
            name = _format_feature_name(d["feature"])
            val = d.get("feature_value")
            val_str = f" (current value: {_format_number(val)})" if val is not None else ""
            contribution = abs(d["shap_value"])
            lines.append(
                f"- **{name.title()}**{val_str}: "
                f"decreases default risk by {contribution:.3f} SHAP units"
            )
        lines.append("")

    return "\n".join(lines)


def generate_counterfactual_summary(changes):
    """
    Generate a natural language summary of counterfactual recommendations.

    Parameters:
        changes: list of dicts (from counterfactual_engine.generate_counterfactuals)
                 each dict maps feature_name -> {original, suggested, change, change_pct}

    Returns:
        str: Recommendation text
    """
    if not changes:
        return "_No actionable counterfactuals were found for this applicant._"

    lines = [
        "### Recommended Changes for Approval",
        "",
        "The following minimal changes could move this applicant to a lower risk tier:",
        "",
    ]

    for i, change_set in enumerate(changes):
        if not change_set:
            continue

        lines.append(f"**Option {i + 1}:**")
        for feature, delta in change_set.items():
            name = _format_feature_name(feature)
            if isinstance(delta, dict) and "original" in delta and "suggested" in delta:
                orig = delta["original"]
                sugg = delta["suggested"]
                if isinstance(orig, (int, float)):
                    is_currency = feature in ["loan_amnt", "annual_inc", "revol_bal", "installment"]
                    is_pct = feature in ["int_rate", "revol_util", "dti"]
                    orig_str = _format_number(orig, is_currency=is_currency, is_pct=is_pct)
                    sugg_str = _format_number(sugg, is_currency=is_currency, is_pct=is_pct)
                    change_dir = "↑" if sugg > orig else "↓"
                    lines.append(f"  - {name.title()}: {orig_str} → {sugg_str} {change_dir}")
                else:
                    lines.append(f"  - {name.title()}: {orig} → {sugg}")
        lines.append("")

    return "\n".join(lines)


def generate_scenario_summary(scenario_results):
    """
    Generate a summary of scenario simulation results.

    Parameters:
        scenario_results: dict from scenario_simulation.simulate_all_scenarios()

    Returns:
        str: Scenario analysis text
    """
    if not scenario_results:
        return "_No scenario simulation results available._"

    lines = [
        "### Stress Test Results",
        "",
        "| Scenario | Baseline | Peak Risk | Risk Increase |",
        "|----------|----------|-----------|---------------|",
    ]

    for name, result in scenario_results.items():
        baseline = result["baseline_probability"]
        peak = result["peak_probability"]
        delta = result["total_risk_increase"]
        lines.append(
            f"| {name} | {baseline:.1%} | {peak:.1%} | "
            f"{'+' if delta >= 0 else ''}{delta:.1%} |"
        )

    lines.append("")

    # Find worst case
    worst = max(scenario_results.items(), key=lambda x: x[1]["peak_probability"])
    worst_name, worst_result = worst

    if worst_result["total_risk_increase"] > 0.1:
        lines.append(
            f"⚠️ **Warning**: Under a **{worst_name}** scenario, this applicant's "
            f"default probability could rise to **{worst_result['peak_probability']:.1%}**, "
            f"an increase of {worst_result['total_risk_increase']:.1%} from baseline."
        )
    else:
        lines.append(
            f"✓ This applicant shows **resilience** across all stress scenarios, "
            f"with a maximum risk increase of {worst_result['total_risk_increase']:.1%}."
        )

    return "\n".join(lines)


def generate_summary(probability, top_drivers=None, counterfactual_changes=None,
                     scenario_results=None):
    """
    Generate a complete decision summary combining all analysis components.

    Parameters:
        probability: float, default probability
        top_drivers: list of SHAP driver dicts
        counterfactual_changes: list of change dicts from DiCE
        scenario_results: dict from scenario simulation

    Returns:
        str: Complete markdown decision summary
    """
    sections = []

    # Risk assessment
    sections.append(generate_risk_assessment(probability, top_drivers or []))

    # Decision recommendation
    tier, emoji = _risk_tier(probability)
    if probability < 0.35:
        sections.append(
            f"**Recommendation**: {emoji} APPROVE — Applicant meets risk tolerance "
            f"thresholds with a {probability:.1%} default probability."
        )
    elif probability < 0.55:
        sections.append(
            f"**Recommendation**: {emoji} REVIEW — Applicant is borderline "
            f"({probability:.1%} default probability). Manual review recommended."
        )
    else:
        sections.append(
            f"**Recommendation**: {emoji} DECLINE — Applicant exceeds risk tolerance "
            f"({probability:.1%} default probability)."
        )

    sections.append("")

    # Counterfactual recommendations (only for high-risk applicants)
    if counterfactual_changes and probability >= 0.35:
        sections.append(generate_counterfactual_summary(counterfactual_changes))

    # Scenario analysis
    if scenario_results:
        sections.append(generate_scenario_summary(scenario_results))

    return "\n".join(sections)


if __name__ == "__main__":
    # Quick demo
    print(generate_summary(
        probability=0.67,
        top_drivers=[
            {"feature": "dti", "shap_value": 0.15, "direction": "increases risk", "feature_value": 42.0},
            {"feature": "credit_history_months", "shap_value": 0.12, "direction": "increases risk", "feature_value": 24},
            {"feature": "annual_inc", "shap_value": -0.08, "direction": "decreases risk", "feature_value": 55000},
        ],
        counterfactual_changes=[
            {"dti": {"original": 42.0, "suggested": 28.0, "change": -14.0, "change_pct": -33.3},
             "annual_inc": {"original": 55000, "suggested": 65000, "change": 10000, "change_pct": 18.2}},
        ],
        scenario_results={
            "Baseline": {"baseline_probability": 0.67, "peak_probability": 0.67, "total_risk_increase": 0.0},
            "Severe Recession": {"baseline_probability": 0.67, "peak_probability": 0.82, "total_risk_increase": 0.15},
        },
    ))
