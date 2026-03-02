"""
Custom evaluators for planner output scoring.

These evaluators are **fully aligned** with the training grader:
    F2 = 5 * (precision * recall) / (4 * precision + recall)

F2 prioritizes recall over precision (β=2), appropriate for retail
where missing a tool causes workflow failure, but extra tools are minor cost.

With enriched datasets, all lookup tools are part of expected_tools,
so no special tolerance is needed.

This ensures evaluation metrics are directly comparable to the training reward.

Supports both JSON structured output and text responses (with fallback).
"""

import json
from typing import Dict, List, Optional

from src.graders.grader import VALID_TOOLS


# =============================================================================
# TOOL EXTRACTION (JSON first, text fallback)
# =============================================================================

def _extract_tools_from_json(response: str) -> Optional[List[str]]:
    """
    Extract tools from JSON structured output.

    Expected format: {"reasoning": "...", "tools": ["tool1", "tool2"]}

    Returns:
        List of lowercase tool names, or None if parsing fails
    """
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict) and "tools" in parsed:
            tools = parsed["tools"]
            if isinstance(tools, list):
                valid_lower = [t.lower() for t in VALID_TOOLS]
                return [t.lower() for t in tools if t.lower() in valid_lower]
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _extract_tools_from_text(response: str) -> List[str]:
    """
    Extract tools via substring matching (fallback for text responses).

    Returns:
        List of lowercase tool names found in the response
    """
    r = response.lower().replace("-", "_")
    return [t.lower() for t in VALID_TOOLS if t.lower() in r]


def extract_predicted_tools(response: str) -> List[str]:
    """
    Extract tools from response (JSON first, text fallback).

    This matches the grader's extraction logic for consistency between
    training reward and evaluation metrics.

    Args:
        response: Model output (JSON string or plain text)

    Returns:
        List of lowercase tool names
    """
    if not response:
        return []

    # Try JSON parsing first (structured output)
    tools = _extract_tools_from_json(response)
    if tools is not None:
        return tools

    # Fallback to text parsing
    return _extract_tools_from_text(response)


# =============================================================================
# EVALUATORS
# =============================================================================

class RecallEvaluator:
    """
    Measures if expected tools are mentioned in the response.

    Supports both JSON structured output and text responses.

    Recall = (number of expected tools found) / (total expected tools)

    Uses SET for deduplication - aligned with training grader.
    """

    def __init__(self):
        pass

    def __call__(self, *, response: str, expected_tools: List[str], **kwargs) -> Dict:
        if not response or not expected_tools:
            return {"recall": 0.0}

        pred_tools = extract_predicted_tools(response)
        pred_set = set(pred_tools)  # Deduplicate - aligned with grader
        expected_set = set(t.lower() for t in expected_tools)

        found = sum(1 for t in expected_set if t in pred_set)
        score = found / len(expected_set)

        return {"recall": round(score, 3)}


class PrecisionEvaluator:
    """
    Measures ratio of correct tools to predicted tools.

    Supports both JSON structured output and text responses.

    Precision = (correct predicted tools) / (total predicted tools)

    Uses SET for deduplication - aligned with training grader.
    """

    def __init__(self):
        pass

    def __call__(self, *, response: str, expected_tools: List[str], **kwargs) -> Dict:
        if not response:
            return {"precision": 0.0}

        pred_tools = extract_predicted_tools(response)
        pred_set = set(pred_tools)  # Deduplicate - aligned with grader
        expected_set = set(t.lower() for t in expected_tools)

        if not pred_set:
            return {"precision": 0.0}

        # Calculate precision: correct / total predicted (using SET)
        correct = sum(1 for t in pred_set if t in expected_set)
        score = correct / len(pred_set)

        return {"precision": round(score, 3)}


class PlannerEvaluator:
    """
    Combined evaluator matching the training grader (F2 score).

    Supports both JSON structured output and text responses.

    Formula: F2 = 5 * (precision * recall) / (4 * precision + recall)
    β=2 means recall is 2x more important than precision.

    Metrics returned:
    - Recall: Finding all expected tools
    - Precision: Not predicting extra unnecessary tools
    - F2: Recall-weighted F-score

    This is aligned with the RFT training grader to ensure evaluation
    metrics are directly comparable to the training reward signal.
    """

    def __init__(self):
        self.recall_eval = RecallEvaluator()
        self.precision_eval = PrecisionEvaluator()

    def __call__(self, *, response: str, expected_tools: List[str], **kwargs) -> Dict:
        # Exclude content-filtered samples from metrics (NaN is ignored by pandas .mean())
        if response == "__CONTENT_FILTER_SKIPPED__":
            return {"recall": float("nan"), "precision": float("nan"), "f2": float("nan")}

        recall = self.recall_eval(response=response, expected_tools=expected_tools)["recall"]
        precision = self.precision_eval(response=response, expected_tools=expected_tools)["precision"]

        # F2 score: recall-weighted F-score (β=2)
        # F2 = (1 + β²) * (precision * recall) / (β² * precision + recall)
        if precision + recall == 0:
            f2 = 0.0
        else:
            beta = 2
            beta_sq = beta ** 2
            f2 = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

        return {
            "recall": recall,
            "precision": precision,
            "f2": round(min(max(f2, 0), 1), 3)
        }


class PlannerEvalWrapper:
    """
    Wrapper for PlannerEvaluator that handles JSON string input.

    Used with azure.ai.evaluation.evaluate() which passes expected_tools as string.
    """

    def __init__(self):
        self.eval = PlannerEvaluator()

    def __call__(self, *, response: str, expected_tools: str, **kwargs) -> Dict:
        import json
        tools = json.loads(expected_tools) if expected_tools else []
        return self.eval(response=response, expected_tools=tools)


# =============================================================================
# STANDALONE EVALUATION FUNCTION (Single Source of Truth)
# =============================================================================

def evaluate_plan(tools_predicted: List[str], tools_expected: List[str]) -> Dict[str, float]:
    """
    Evaluate planning quality using metrics aligned with RFT training grader.

    This is the SINGLE SOURCE OF TRUTH for evaluation metrics.
    All notebooks and modules should use this function.

    Metrics:
    - Recall: % of expected tools that were predicted
    - Precision: % of predicted tools that were correct
    - F2: Recall-weighted F-score (β=2)

    Formula: F2 = 5 * (precision * recall) / (4 * precision + recall)

    Uses SET for deduplication - aligned with training grader.

    Args:
        tools_predicted: List of tools predicted by the planner
        tools_expected: List of expected tools (ground truth)

    Returns:
        Dict with recall, precision, f2 scores
    """
    if not tools_expected:
        return {"recall": 1.0, "precision": 1.0, "f2": 1.0}

    pred_set = set(t.lower() for t in tools_predicted)
    exp_set = set(t.lower() for t in tools_expected)

    # Recall: % of expected tools found
    found = sum(1 for t in exp_set if t in pred_set)
    recall = found / len(exp_set)

    # Precision: % of predicted tools that are correct
    if not pred_set:
        precision = 0.0
    else:
        correct = sum(1 for t in pred_set if t in exp_set)
        precision = correct / len(pred_set)

    # F2: Recall-weighted F-score (β=2)
    if precision + recall == 0:
        f2 = 0.0
    else:
        beta = 2
        beta_sq = beta ** 2
        f2 = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "f2": round(f2, 3)
    }


def test_evaluators():
    """Quick test to verify evaluators work correctly with both JSON and text."""
    evaluator = PlannerEvaluator()

    # Test with text response
    text_result = evaluator(
        response="First use find_user_id_by_name_zip, then get_order_details",
        expected_tools=["find_user_id_by_name_zip", "get_order_details"]
    )
    print(f"Text response: {text_result}")
    assert text_result["f2"] == 1.0, f"Text F2 should be 1.0, got {text_result['f2']}"

    # Test with JSON structured output
    json_response = json.dumps({
        "reasoning": "Customer needs to find user first then view orders",
        "tools": ["find_user_id_by_name_zip", "get_order_details"]
    })
    json_result = evaluator(
        response=json_response,
        expected_tools=["find_user_id_by_name_zip", "get_order_details"]
    )
    print(f"JSON response: {json_result}")
    assert json_result["f2"] == 1.0, f"JSON F2 should be 1.0, got {json_result['f2']}"

    # Test extract_predicted_tools directly
    assert extract_predicted_tools("") == []
    assert "get_order_details" in extract_predicted_tools("Use get_order_details")
    assert extract_predicted_tools(json_response) == ["find_user_id_by_name_zip", "get_order_details"]

    print("All evaluator tests passed!")
    return json_result
