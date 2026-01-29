"""
Azure AI Evaluation SDK - Agent Evaluators.

This module provides wrappers for Azure AI Evaluation SDK's agent-specific evaluators:
- TaskAdherenceEvaluator: Does the final response address the user's request?
- IntentResolutionEvaluator: Was the user's intent correctly understood?
- ToolCallAccuracyEvaluator: Were tools called with correct arguments?

Our multi-agent architecture separates concerns:
- Planner: predicts WHICH tools to call → measured by F2/Recall/Precision
- Sub-agents: extract arguments and execute → measured by ToolCallAccuracy

TaskAdherence and IntentResolution evaluate the END result (final response quality).
ToolCallAccuracy evaluates the PROCESS (were arguments extracted correctly?).

Reference: https://learn.microsoft.com/azure/ai-foundry/how-to/develop/agent-evaluate-sdk
"""

import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from src.settings import (
    AZURE_ENDPOINT,
    AZURE_API_KEY,
    EVAL_DEPLOYMENT,  # gpt-4.1-mini (fast, non-reasoning)
    AZURE_API_VERSION,
    load_tool_definitions
)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

def get_model_config() -> Dict[str, str]:
    """
    Get Azure OpenAI model configuration for LLM-as-a-judge evaluators.

    Uses gpt-4.1-mini (non-reasoning model) as the judge model for faster evaluation.
    Non-reasoning models use is_reasoning_model=False in evaluator init.

    Returns:
        Dict with azure_endpoint, api_key, azure_deployment, api_version
    """
    return {
        "azure_endpoint": AZURE_ENDPOINT,
        "api_key": AZURE_API_KEY,
        "azure_deployment": EVAL_DEPLOYMENT,  # gpt-4.1-mini
        "api_version": AZURE_API_VERSION
    }


# Flag for reasoning models (o-series) - gpt-4.1-mini is NOT a reasoning model
IS_REASONING_MODEL = False


# =============================================================================
# MESSAGE FORMAT CONVERSION
# =============================================================================

def convert_workflow_trace_to_agent_messages(
    execution_trace: List[Dict],
    final_response: str,
    workflow_id: str = None
) -> List[Dict]:
    """
    Convert workflow execution trace to Azure AI Evaluation agent message format.

    Uses the FLAT format documented in "Evaluate other agents" section:
    https://learn.microsoft.com/azure/ai-foundry/how-to/develop/agent-evaluate-sdk#agent-message-schema

    Args:
        execution_trace: List of tool execution results from run_workflow()
        final_response: The synthesizer's final response
        workflow_id: Optional ID for the workflow run

    Returns:
        List of messages in Azure AI Evaluation format
    """
    if workflow_id is None:
        workflow_id = f"run_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    messages = []

    for idx, entry in enumerate(execution_trace):
        timestamp = datetime.utcnow().isoformat() + "Z"
        tool_name = entry.get("tool_called") or entry.get("tool_name") or entry.get("tool", "unknown")
        tool_call_id = f"call_{tool_name}_{idx}"

        # Agent makes a tool call - FLAT format required by Azure AI Evaluation SDK
        # Reference: https://learn.microsoft.com/azure/ai-foundry/how-to/develop/agent-evaluate-sdk#agent-message-schema
        messages.append({
            "createdAt": timestamp,
            "run_id": workflow_id,
            "role": "assistant",
            "content": [{
                "type": "tool_call",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "arguments": entry.get("arguments", {})
            }]
        })

        # Tool returns result
        tool_result = entry.get("result") if entry.get("result") else entry.get("error", "")
        messages.append({
            "createdAt": timestamp,
            "run_id": workflow_id,
            "tool_call_id": tool_call_id,
            "role": "tool",
            "content": [{
                "type": "tool_result",
                "tool_result": str(tool_result)
            }]
        })

    # Final assistant response (from synthesizer)
    if final_response:
        messages.append({
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "run_id": workflow_id,
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": final_response
            }]
        })

    return messages


def convert_tool_definitions_for_evaluator(tool_definitions: List[Dict]) -> List[Dict]:
    """
    Convert tool definitions to the format expected by Azure evaluators.

    The evaluator expects (per Azure AI Evaluation SDK docs):
    {
        "id": "tool_name",      # Required for ToolCallAccuracyEvaluator
        "name": "tool_name",
        "description": "...",
        "parameters": {...}
    }

    Args:
        tool_definitions: List of OpenAI function tool definitions

    Returns:
        List of tool definitions in Azure evaluator format
    """
    converted = []
    for tool in tool_definitions:
        if "function" in tool:
            func = tool["function"]
            name = func.get("name", "")
            converted.append({
                "id": name,  # Required by ToolCallAccuracyEvaluator
                "name": name,
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {})
            })
        else:
            # Already in simple format - ensure id exists
            name = tool.get("name", "")
            if "id" not in tool:
                tool = {**tool, "id": name}
            converted.append(tool)
    return converted


# =============================================================================
# EVALUATOR WRAPPERS
# =============================================================================

def evaluate_task_adherence(
    query: List[Dict],
    response: List[Dict],
    tool_definitions: List[Dict] = None,
    threshold: int = 3
) -> Dict[str, Any]:
    """
    Evaluate task adherence using Azure AI Evaluation SDK.

    Uses agent message format for both query and response.

    Args:
        query: Agent message list with system message + user message
        response: Agent messages (tool_calls, tool_results, final text)
        tool_definitions: Optional tool definitions for context
        threshold: Pass/fail threshold (1-5 scale, default 3)

    Returns:
        Dict with task_adherence (1-5), result (pass/fail), reason
    """
    try:
        from azure.ai.evaluation import TaskAdherenceEvaluator, AzureOpenAIModelConfiguration
    except ImportError:
        return {
            "task_adherence": None,
            "task_adherence_result": "error",
            "error": "azure-ai-evaluation not installed. Run: pip install azure-ai-evaluation"
        }

    model_config = AzureOpenAIModelConfiguration(**get_model_config())

    evaluator = TaskAdherenceEvaluator(
        model_config=model_config,
        threshold=threshold,
        is_reasoning_model=IS_REASONING_MODEL
    )

    kwargs = {"query": query, "response": response}
    if tool_definitions:
        kwargs["tool_definitions"] = convert_tool_definitions_for_evaluator(tool_definitions)

    try:
        return evaluator(**kwargs)
    except Exception as e:
        # Handle Azure content filter errors (jailbreak detection, etc.)
        error_msg = str(e)
        if "content_filter" in error_msg or "content_management_policy" in error_msg:
            return {
                "task_adherence": None,
                "task_adherence_result": "skipped",
                "error": "content_filter_triggered"
            }
        raise


def evaluate_intent_resolution(
    query: List[Dict],
    response: List[Dict],
    tool_definitions: List[Dict] = None,
    threshold: int = 3
) -> Dict[str, Any]:
    """
    Evaluate intent resolution using Azure AI Evaluation SDK.

    Uses agent message format for both query and response.

    Args:
        query: Agent message list with system message + user message
        response: Agent messages in OpenAI format
        tool_definitions: Optional tool definitions for context
        threshold: Pass/fail threshold (1-5 scale, default 3)

    Returns:
        Dict with intent_resolution (1-5), result (pass/fail), reason
    """
    try:
        from azure.ai.evaluation import IntentResolutionEvaluator, AzureOpenAIModelConfiguration
    except ImportError:
        return {
            "intent_resolution": None,
            "intent_resolution_result": "error",
            "error": "azure-ai-evaluation not installed. Run: pip install azure-ai-evaluation"
        }

    model_config = AzureOpenAIModelConfiguration(**get_model_config())

    evaluator = IntentResolutionEvaluator(
        model_config=model_config,
        threshold=threshold,
        is_reasoning_model=IS_REASONING_MODEL
    )

    kwargs = {"query": query, "response": response}
    if tool_definitions:
        kwargs["tool_definitions"] = convert_tool_definitions_for_evaluator(tool_definitions)

    try:
        return evaluator(**kwargs)
    except Exception as e:
        # Handle Azure content filter errors (jailbreak detection, etc.)
        error_msg = str(e)
        if "content_filter" in error_msg or "content_management_policy" in error_msg:
            return {
                "intent_resolution": None,
                "intent_resolution_result": "skipped",
                "error": "content_filter_triggered"
            }
        raise


def evaluate_tool_call_accuracy(
    query: List[Dict],
    response: List[Dict],
    tool_definitions: List[Dict] = None,
    threshold: int = 3
) -> Dict[str, Any]:
    """
    Evaluate tool call accuracy using Azure AI Evaluation SDK.

    This evaluator assesses whether tools were called with correct arguments,
    which measures the quality of sub-agent argument extraction.

    Args:
        query: Agent message list with system message + user message
        response: Agent messages (tool_calls, tool_results, final text)
        tool_definitions: Tool definitions (REQUIRED for this evaluator)
        threshold: Pass/fail threshold (1-5 scale, default 3)

    Returns:
        Dict with tool_call_accuracy (1-5), result (pass/fail), reason
    """
    try:
        from azure.ai.evaluation import ToolCallAccuracyEvaluator, AzureOpenAIModelConfiguration
    except ImportError:
        return {
            "tool_call_accuracy": None,
            "tool_call_accuracy_result": "error",
            "error": "azure-ai-evaluation not installed. Run: pip install azure-ai-evaluation"
        }

    model_config = AzureOpenAIModelConfiguration(**get_model_config())

    evaluator = ToolCallAccuracyEvaluator(
        model_config=model_config,
        threshold=threshold,
        is_reasoning_model=IS_REASONING_MODEL
    )

    # Tool definitions are required for ToolCallAccuracy
    if tool_definitions is None:
        tool_definitions = load_tool_definitions()

    kwargs = {
        "query": query,
        "response": response,
        "tool_definitions": convert_tool_definitions_for_evaluator(tool_definitions)
    }

    try:
        return evaluator(**kwargs)
    except Exception as e:
        # Handle Azure content filter errors (jailbreak detection, etc.)
        error_msg = str(e)
        if "content_filter" in error_msg or "content_management_policy" in error_msg:
            return {
                "tool_call_accuracy": None,
                "tool_call_accuracy_result": "skipped",
                "error": "content_filter_triggered"
            }
        raise


# =============================================================================
# COMBINED EVALUATION
# =============================================================================

def evaluate_workflow_result(
    user_request: str,
    execution_trace: List[Dict],
    final_response: str,
    tool_definitions: List[Dict] = None,
    system_message: str = None
) -> Dict[str, Any]:
    """
    Evaluate a workflow result using Azure AI Evaluation SDK.

    Runs three agent evaluators:
    1. TaskAdherenceEvaluator - Does the response address the user's request?
    2. IntentResolutionEvaluator - Was the user's intent correctly understood?
    3. ToolCallAccuracyEvaluator - Were tools called with correct arguments?

    Architecture note:
    - Planner: predicts WHICH tools → measured by F2/Recall/Precision
    - Sub-agents: extract arguments → measured by ToolCallAccuracy

    Args:
        user_request: The original user request
        execution_trace: List of tool execution results from workflow
        final_response: The synthesizer's final response
        tool_definitions: Optional, defaults to load_tool_definitions()
        system_message: Optional system message for context

    Returns:
        Dict with evaluation results and summary
    """
    # Load tool definitions if not provided
    if tool_definitions is None:
        tool_definitions = load_tool_definitions()

    # Convert to agent message format for TaskAdherence and IntentResolution
    agent_messages = convert_workflow_trace_to_agent_messages(
        execution_trace=execution_trace,
        final_response=final_response
    )

    # Build query in agent message format for TaskAdherence/IntentResolution
    # Reference: https://learn.microsoft.com/azure/ai-foundry/how-to/develop/agent-evaluate-sdk
    system_content = system_message or "You are a helpful retail customer service assistant."
    query_messages = [
        {"role": "system", "content": system_content},
        {
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "role": "user",
            "content": [{"type": "text", "text": user_request}]
        }
    ]

    # Run evaluators
    results = {}

    # TaskAdherence: query and response as agent messages
    results["task_adherence"] = evaluate_task_adherence(
        query=query_messages,
        response=agent_messages,
        tool_definitions=tool_definitions
    )

    # IntentResolution: query and response as agent messages
    results["intent_resolution"] = evaluate_intent_resolution(
        query=query_messages,
        response=agent_messages,
        tool_definitions=tool_definitions
    )

    # ToolCallAccuracy: measures sub-agent argument extraction quality
    results["tool_call_accuracy"] = evaluate_tool_call_accuracy(
        query=query_messages,
        response=agent_messages,
        tool_definitions=tool_definitions
    )

    # Add summary with safe float conversion
    scores = []
    passed = 0
    evaluator_keys = ["task_adherence", "intent_resolution", "tool_call_accuracy"]

    for key in evaluator_keys:
        result = results.get(key, {})
        score = result.get(key)
        if score is not None:
            # SDK may return scores as strings or "not applicable"
            try:
                scores.append(float(score))
            except (ValueError, TypeError):
                pass  # Skip non-numeric scores like "not applicable"
        if result.get(f"{key}_result") == "pass":
            passed += 1

    results["summary"] = {
        "average_score": round(sum(scores) / len(scores), 2) if scores else None,
        "pass_count": passed,
        "total_evaluators": 3,
        "all_passed": passed == 3
    }

    return results


def print_evaluation_results(results: Dict[str, Any]) -> None:
    """
    Pretty print evaluation results.

    Args:
        results: Output from evaluate_workflow_result()
    """
    print("\n" + "=" * 60)
    print("AGENT QUALITY METRICS")
    print("=" * 60)

    # Task Adherence (binary: 0=fail, 1=pass)
    ta = results.get("task_adherence", {})
    score = ta.get("task_adherence", "N/A")
    status = ta.get("task_adherence_result", "N/A")
    # TaskAdherence returns binary score (0 or 1), not 1-5 scale
    score_display = "PASS" if score == 1.0 else "FAIL" if score == 0.0 else str(score)
    print(f"\nTask Adherence: {score_display} [{status.upper()}]")
    if "task_adherence_reason" in ta:
        reason = ta["task_adherence_reason"][:100] + "..." if len(ta.get("task_adherence_reason", "")) > 100 else ta.get("task_adherence_reason", "")
        print(f"   Reason: {reason}")

    # Intent Resolution
    ir = results.get("intent_resolution", {})
    score = ir.get("intent_resolution", "N/A")
    status = ir.get("intent_resolution_result", "N/A")
    print(f"\nIntent Resolution: {score}/5 [{status.upper()}]")
    if "intent_resolution_reason" in ir:
        reason = ir["intent_resolution_reason"][:100] + "..." if len(ir.get("intent_resolution_reason", "")) > 100 else ir.get("intent_resolution_reason", "")
        print(f"   Reason: {reason}")

    # Tool Call Accuracy
    tca = results.get("tool_call_accuracy", {})
    score = tca.get("tool_call_accuracy", "N/A")
    status = tca.get("tool_call_accuracy_result", "N/A")
    print(f"\nTool Call Accuracy: {score}/5 [{status.upper()}]")
    if "tool_call_accuracy_reason" in tca:
        reason = tca["tool_call_accuracy_reason"][:100] + "..." if len(tca.get("tool_call_accuracy_reason", "")) > 100 else tca.get("tool_call_accuracy_reason", "")
        print(f"   Reason: {reason}")

    # Summary
    summary = results.get("summary", {})
    print(f"\n{'─' * 60}")
    print(f"Summary:")
    print(f"   Passed: {summary.get('pass_count', 0)}/{summary.get('total_evaluators', 3)}")
    status = "ALL PASSED" if summary.get("all_passed") else "SOME FAILED"
    print(f"   Status: {status}")
    print("=" * 60 + "\n")
