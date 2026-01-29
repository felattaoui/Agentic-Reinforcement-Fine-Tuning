"""
Evaluation functions for multi-agent workflow.

Uses the SINGLE SOURCE OF TRUTH from src.evaluation.evaluators.evaluate_plan().

F2 = 5 * (precision * recall) / (4 * precision + recall)

F2 prioritizes recall over precision (β=2), appropriate for retail
where missing a tool causes workflow failure, but extra tools are minor cost.
"""

# Re-export evaluate_plan from single source of truth (aligned with training grader)
from src.evaluation.evaluators import evaluate_plan

__all__ = ["evaluate_plan"]
