"""
Planner Grader for RFT Training.

This grader scores model outputs using F2 score with a penalty for extra tools.
F2 prioritizes recall over precision (β=2), appropriate for retail customer service
where missing a tool causes workflow failure.

Formula: F2 = 5 * (precision * recall) / (4 * precision + recall)
Penalty: -3% per extra tool predicted (to discourage over-prediction)
Final Score: max(0, F2 - penalty)

Note: With enriched datasets, lookup tools are part of expected_tools,
so no special tolerance is needed.

The grader code is also available as a string (GRADER_CODE) for Azure OpenAI.
"""

import json
from typing import List

# Valid tools from tau-bench retail dataset
VALID_TOOLS = [
    "calculate",
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "list_all_product_types",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "transfer_to_human_agents"
]


def _extract_tools_from_text(response: str) -> List[str]:
    """
    Extract tool names from text using substring matching (legacy method).

    Args:
        response: The text response to parse

    Returns:
        List of tool names found in order of appearance
    """
    pred_tools = []
    for tool in VALID_TOOLS:
        if tool in response:
            pos = response.find(tool)
            pred_tools.append((pos, tool))
    pred_tools.sort(key=lambda x: x[0])
    return [t[1] for t in pred_tools]


def _extract_tools_from_json(output_text: str) -> List[str]:
    """
    Extract tool names from JSON structured output.

    Args:
        output_text: JSON string with 'tools' field

    Returns:
        List of tool names, or None if parsing fails
    """
    try:
        parsed = json.loads(output_text)
        if isinstance(parsed, dict) and "tools" in parsed:
            tools = parsed["tools"]
            if isinstance(tools, list):
                # Validate tools are from VALID_TOOLS
                return [t.lower() for t in tools if t.lower() in [v.lower() for v in VALID_TOOLS]]
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def grade(sample: dict, item: dict) -> float:
    """
    Grade a planner response using F2 score (recall-weighted F-score).

    F2 prioritizes recall over precision (β=2), appropriate for retail
    where missing a tool causes failure, but extra tools are minor cost.

    Supports both structured JSON output and text-based output (fallback).
    For JSON: expects {"reasoning": "...", "tools": ["tool1", "tool2"]}
    For text: uses substring matching to find tool names

    Args:
        sample: Dict containing 'output_text' with the model's response
        item: Dict containing 'reference_answer' with expected tools

    Returns:
        float: F2 score between 0.0 and 1.0
    """
    # ==========================================================================
    # PARSE REFERENCE (expected tools)
    # ==========================================================================
    try:
        ref_raw = item.get("reference_answer", {})
        reference = json.loads(ref_raw) if isinstance(ref_raw, str) else ref_raw
        expected_tools = reference.get("expected_tools", [])
    except:
        return 0.0

    # ==========================================================================
    # EXTRACT PREDICTED TOOLS (JSON first, then text fallback)
    # ==========================================================================
    output_text = sample.get("output_text", "") or ""

    # Try JSON parsing first (structured output)
    pred_lower = _extract_tools_from_json(output_text)

    # Fallback to text parsing if JSON fails
    if pred_lower is None:
        response = output_text.lower().replace("-", "_")
        pred_lower = _extract_tools_from_text(response)

    # ==========================================================================
    # EDGE CASES
    # ==========================================================================
    if not expected_tools:
        return 1.0 if len(pred_lower) == 0 else 0.0

    if not output_text or len(pred_lower) == 0:
        return 0.0

    exp_lower = [t.lower() for t in expected_tools]
    pred_set = set(pred_lower)
    exp_set = set(exp_lower)

    # ==========================================================================
    # RECALL: % of expected tools found
    # ==========================================================================
    found = sum(1 for t in exp_set if t in pred_set)
    recall = found / len(exp_set)

    # ==========================================================================
    # PRECISION: ratio of correct tools to predicted tools
    # ==========================================================================
    if len(pred_set) == 0:
        precision = 0.0
    else:
        correct = sum(1 for t in pred_set if t in exp_set)
        precision = correct / len(pred_set)

    # ==========================================================================
    # F2 SCORE: recall-weighted F-score (β=2)
    # F2 = (1 + β²) * (precision * recall) / (β² * precision + recall)
    # With β=2: F2 = 5 * (precision * recall) / (4 * precision + recall)
    # ==========================================================================
    if precision + recall == 0:
        return 0.0

    beta = 2
    beta_sq = beta ** 2
    f2 = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

    # ==========================================================================
    # PENALTY: 3% per extra tool to discourage over-prediction
    # ==========================================================================
    extra_tools = len(pred_set - exp_set)
    penalty = extra_tools * 0.03
    final_score = f2 - penalty

    return min(max(final_score, 0.0), 1.0)


# =============================================================================
# RAW CODE STRING FOR AZURE OPENAI
# =============================================================================
# This is the exact code that will be sent to Azure OpenAI for RFT training.
# It must be self-contained (no imports from src/).

GRADER_CODE = '''
import json

def grade(sample, item):
    """
    Grade a planner response using F2 score (recall-weighted F-score).

    F2 prioritizes recall over precision (β=2), appropriate for retail
    where missing a tool causes failure, but extra tools are minor cost.

    Supports both structured JSON output and text-based output (fallback).
    For JSON: expects {"reasoning": "...", "tools": ["tool1", "tool2"]}
    For text: uses substring matching to find tool names
    """

    valid_tools = [
        "calculate", "cancel_pending_order", "exchange_delivered_order_items",
        "find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details",
        "get_product_details", "get_user_details", "list_all_product_types",
        "modify_pending_order_address", "modify_pending_order_items",
        "modify_pending_order_payment", "modify_user_address",
        "return_delivered_order_items", "transfer_to_human_agents"
    ]
    valid_tools_lower = [t.lower() for t in valid_tools]

    # ==========================================================================
    # PARSE REFERENCE (expected tools)
    # ==========================================================================
    try:
        ref_raw = item.get("reference_answer", {})
        reference = json.loads(ref_raw) if isinstance(ref_raw, str) else ref_raw
        expected_tools = reference.get("expected_tools", [])
    except:
        return 0.0

    # ==========================================================================
    # EXTRACT PREDICTED TOOLS (JSON first, then text fallback)
    # ==========================================================================
    output_text = sample.get("output_text", "") or ""
    pred_lower = None

    # Try JSON parsing first (structured output)
    try:
        parsed = json.loads(output_text)
        if isinstance(parsed, dict) and "tools" in parsed:
            tools = parsed["tools"]
            if isinstance(tools, list):
                pred_lower = [t.lower() for t in tools if t.lower() in valid_tools_lower]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback to text parsing if JSON fails
    if pred_lower is None:
        response = output_text.lower().replace("-", "_")
        pred_tools = []
        for tool in valid_tools:
            if tool in response:
                pos = response.find(tool)
                pred_tools.append((pos, tool))
        pred_tools.sort(key=lambda x: x[0])
        pred_lower = [t[1] for t in pred_tools]

    # ==========================================================================
    # EDGE CASES
    # ==========================================================================
    if not expected_tools:
        return 1.0 if len(pred_lower) == 0 else 0.0

    if not output_text or len(pred_lower) == 0:
        return 0.0

    exp_lower = [t.lower() for t in expected_tools]
    pred_set = set(pred_lower)
    exp_set = set(exp_lower)

    # ==========================================================================
    # RECALL: % of expected tools found
    # ==========================================================================
    found = sum(1 for t in exp_set if t in pred_set)
    recall = found / len(exp_set)

    # ==========================================================================
    # PRECISION: ratio of correct tools to predicted tools
    # ==========================================================================
    if len(pred_set) == 0:
        precision = 0.0
    else:
        correct = sum(1 for t in pred_set if t in exp_set)
        precision = correct / len(pred_set)

    # ==========================================================================
    # F2 SCORE: recall-weighted F-score (β=2)
    # F2 = (1 + β²) * (precision * recall) / (β² * precision + recall)
    # With β=2: F2 = 5 * (precision * recall) / (4 * precision + recall)
    # ==========================================================================
    if precision + recall == 0:
        return 0.0

    beta = 2
    beta_sq = beta ** 2
    f2 = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

    # ==========================================================================
    # PENALTY: 3% per extra tool to discourage over-prediction
    # ==========================================================================
    extra_tools = len(pred_set - exp_set)
    penalty = extra_tools * 0.03
    final_score = f2 - penalty

    return min(max(final_score, 0.0), 1.0)
'''
