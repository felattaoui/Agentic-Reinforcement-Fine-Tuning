"""
Evaluation module for comparing baseline vs fine-tuned models.

This module provides:
- Evaluators: Recall, Precision, and combined PlannerEvaluator (F1 score)
- Deployment management: list, deploy, delete Azure OpenAI deployments
- Content filter: Create RAI policies for evaluation
- Response generation: Async parallel response generation
"""

from src.evaluation.evaluators import (
    RecallEvaluator,
    PrecisionEvaluator,
    PlannerEvaluator
)
from src.evaluation.deployment import (
    get_azure_credentials,
    list_finetuned_deployments,
    deploy_model,
    delete_deployment,
    check_if_deployed
)
from src.evaluation.content_filter import create_no_jailbreak_filter
from src.evaluation.generate import generate_responses_async

__all__ = [
    # Evaluators
    "RecallEvaluator",
    "PrecisionEvaluator",
    "PlannerEvaluator",
    # Deployment
    "get_azure_credentials",
    "list_finetuned_deployments",
    "deploy_model",
    "delete_deployment",
    "check_if_deployed",
    # Content filter
    "create_no_jailbreak_filter",
    # Generation
    "generate_responses_async",
]
