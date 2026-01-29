"""
Cost calculation utilities for Azure OpenAI models.

Supports:
- Inference costs (input/output/reasoning tokens)
- Fine-tuning costs (training runs, hosting)
- Amortization over time
- Break-even analysis
"""

from typing import Dict, Optional


def calculate_model_cost(
    usage: Dict,
    input_price: float,
    output_price: float,
    num_samples: int = 1,
    # Fine-tuning parameters (optional)
    training_runs: int = 0,
    training_hours_per_run: float = 1.0,
    training_rate: float = 100.0,
    hosting_hours_monthly: float = 720,  # 24*30
    hosting_rate: float = 1.70,
    amortization_months: int = 6,
    # Volume projection
    monthly_requests: int = 0
) -> Dict:
    """
    Calculate total cost for a model including inference and fine-tuning costs.

    Args:
        usage: Token usage dict with input_tokens, output_tokens, reasoning_tokens
        input_price: Price per 1M input tokens (USD)
        output_price: Price per 1M output tokens (USD)
        num_samples: Number of samples in the evaluation (for extrapolation)
        training_runs: Number of fine-tuning training runs (experiments)
        training_hours_per_run: Hours per training run
        training_rate: Cost per hour for training (USD)
        hosting_hours_monthly: Hours of hosting per month (720 = 24/7)
        hosting_rate: Cost per hour for hosting (USD)
        amortization_months: Months to amortize training costs over
        monthly_requests: Projected monthly request volume (for total cost)

    Returns:
        Dict with cost breakdown:
            - inference_cost_eval: Cost for the evaluation run
            - inference_cost_per_1k: Cost per 1,000 requests
            - training_cost_total: Total training cost (all runs)
            - hosting_cost_monthly: Monthly hosting cost
            - fixed_cost_amortized_monthly: Training cost amortized per month
            - total_fixed_monthly: Total fixed costs per month (hosting + amortized training)
            - total_monthly: Total monthly cost (fixed + inference) if monthly_requests > 0
    """
    # Inference cost calculation
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    reasoning_tokens = usage.get("reasoning_tokens", 0)

    input_cost = input_tokens * input_price / 1_000_000
    # Reasoning tokens are billed at output rate
    output_cost = (output_tokens + reasoning_tokens) * output_price / 1_000_000
    inference_cost_eval = input_cost + output_cost

    # Extrapolate to per 1,000 requests
    inference_cost_per_1k = inference_cost_eval * (1000 / num_samples) if num_samples > 0 else 0

    # Fine-tuning costs
    training_cost_total = training_runs * training_hours_per_run * training_rate
    hosting_cost_monthly = hosting_hours_monthly * hosting_rate

    # Amortized training cost per month
    fixed_cost_amortized_monthly = training_cost_total / amortization_months if amortization_months > 0 else 0

    # Total fixed monthly cost
    total_fixed_monthly = hosting_cost_monthly + fixed_cost_amortized_monthly

    # Total monthly cost with projected volume
    total_monthly = None
    if monthly_requests > 0:
        inference_monthly = inference_cost_per_1k * (monthly_requests / 1000)
        total_monthly = total_fixed_monthly + inference_monthly

    return {
        "inference_cost_eval": round(inference_cost_eval, 4),
        "inference_cost_per_1k": round(inference_cost_per_1k, 2),
        "training_cost_total": round(training_cost_total, 2),
        "hosting_cost_monthly": round(hosting_cost_monthly, 2),
        "fixed_cost_amortized_monthly": round(fixed_cost_amortized_monthly, 2),
        "total_fixed_monthly": round(total_fixed_monthly, 2),
        "total_monthly": round(total_monthly, 2) if total_monthly else None
    }


def calculate_breakeven(
    finetuned_cost_per_1k: float,
    alternative_cost_per_1k: float,
    training_cost_total: float,
    hosting_cost_monthly: float,
    amortization_months: int = 6
) -> Dict:
    """
    Calculate break-even point for fine-tuning vs alternative model.

    Args:
        finetuned_cost_per_1k: Inference cost per 1k requests for fine-tuned model
        alternative_cost_per_1k: Inference cost per 1k requests for alternative
        training_cost_total: Total training cost (all runs)
        hosting_cost_monthly: Monthly hosting cost
        amortization_months: Months to amortize training costs

    Returns:
        Dict with break-even analysis:
            - savings_per_1k: Savings per 1,000 requests
            - savings_per_request: Savings per single request
            - breakeven_month1: Requests needed in month 1 (training + hosting)
            - breakeven_monthly: Requests needed per month (hosting only)
            - breakeven_daily: Daily request threshold
            - is_viable: True if fine-tuning saves money on inference
    """
    savings_per_1k = alternative_cost_per_1k - finetuned_cost_per_1k
    savings_per_request = savings_per_1k / 1000 if savings_per_1k != 0 else 0

    if savings_per_1k <= 0:
        return {
            "savings_per_1k": round(savings_per_1k, 2),
            "savings_per_request": round(savings_per_request, 6),
            "breakeven_month1": None,
            "breakeven_monthly": None,
            "breakeven_daily": None,
            "is_viable": False,
            "message": "Fine-tuned model is more expensive per request"
        }

    # Month 1: cover training (amortized) + hosting
    amortized_training = training_cost_total / amortization_months
    month1_fixed = amortized_training + hosting_cost_monthly
    breakeven_month1 = (month1_fixed / savings_per_1k) * 1000

    # Ongoing: just hosting
    breakeven_monthly = (hosting_cost_monthly / savings_per_1k) * 1000
    breakeven_daily = breakeven_monthly / 30

    return {
        "savings_per_1k": round(savings_per_1k, 2),
        "savings_per_request": round(savings_per_request, 6),
        "breakeven_month1": round(breakeven_month1),
        "breakeven_monthly": round(breakeven_monthly),
        "breakeven_daily": round(breakeven_daily),
        "is_viable": True
    }
