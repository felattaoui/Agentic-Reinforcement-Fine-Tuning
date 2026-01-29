"""
Multi-agent workflow orchestration with ReAct pattern.

This module provides the main workflow that:
1. Gets a plan from the Planner (fine-tuned model)
2. Parses tool names from the plan
3. Creates an ExecutorAgent with only the predicted tools
4. Executes via ReAct pattern (iterates automatically)
5. Collects results and generates final response

The ExecutorAgent uses Agent Framework's native ReAct loop:
- Single agent.run() call
- Agent iterates: tool call → observe → decide → repeat
- Continues until task is complete
"""

import json
import time
from typing import List, Dict, Any, Optional

from src.multiagent.tools import (
    reset_tool_log,
    get_tool_log,
    set_active_database,
    KNOWN_TOOLS
)
from src.multiagent.database import RetailDatabase, load_database


def _extract_tools_from_text(plan_text: str) -> List[str]:
    """
    Extract tool names from text using substring matching (legacy method).

    Args:
        plan_text: The raw text output from the Planner

    Returns:
        List of tool names in order of appearance
    """
    text_normalized = plan_text.lower().replace("-", "_").replace(" ", "_")

    found_with_positions = []
    for tool in KNOWN_TOOLS:
        pos = text_normalized.find(tool.lower())
        if pos >= 0:
            found_with_positions.append((tool, pos))

    # Sort by position (order of appearance)
    found_with_positions.sort(key=lambda x: x[1])

    # Remove duplicates while preserving order
    seen = set()
    result = []
    for tool, _ in found_with_positions:
        if tool not in seen:
            seen.add(tool)
            result.append(tool)

    return result


def extract_tools_from_plan(plan_response) -> List[str]:
    """
    Extract tool names from the Planner's response.

    Supports multiple output formats:
    1. Pydantic PlannerResponse object (from Agent Framework with response_format)
    2. JSON text: {"tools": ["tool1", "tool2"]}
    3. Plain text: substring matching to find tool names (fallback)

    Args:
        plan_response: One of:
            - PlannerResponse object (has .tools attribute)
            - Response object with .text attribute (JSON or plain text)
            - String (JSON or plain text)

    Returns:
        List of tool names
    """
    known_tools_lower = {t.lower() for t in KNOWN_TOOLS}

    # Case 1: Pydantic PlannerResponse (has .tools attribute directly)
    if hasattr(plan_response, 'tools') and isinstance(plan_response.tools, list):
        return [t for t in plan_response.tools if t.lower() in known_tools_lower]

    # Case 2: Response object with .text attribute
    if hasattr(plan_response, 'text'):
        plan_text = plan_response.text
    else:
        plan_text = str(plan_response) if plan_response else ""

    # Try JSON parsing (structured output as text)
    try:
        parsed = json.loads(plan_text)
        if isinstance(parsed, dict) and "tools" in parsed:
            tools = parsed["tools"]
            if isinstance(tools, list):
                return [t for t in tools if t.lower() in known_tools_lower]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback to text parsing
    return _extract_tools_from_text(plan_text)


async def run_planner_only(
    user_request: str,
    planner
) -> Dict[str, Any]:
    """
    Run only the Planner step (no execution).

    Use this for evaluating Planner tool selection without executing tools.
    This is faster and doesn't require database setup.

    Args:
        user_request: Customer's natural language request
        planner: Planner agent (fine-tuned model)

    Returns:
        Dict with plan_text and tools_planned
    """
    # Fresh thread for Planner to avoid context contamination
    planner_thread = planner.get_new_thread()
    plan_result = await planner.run(user_request, thread=planner_thread)

    # Handle both structured output (Pydantic) and text output
    if hasattr(plan_result, 'value') and plan_result.value is not None:
        tools_planned = extract_tools_from_plan(plan_result.value)
        plan_text = json.dumps({"tools": tools_planned})
    else:
        plan_text = plan_result.text
        tools_planned = extract_tools_from_plan(plan_result.text)

    return {
        "plan_text": plan_text,
        "tools_planned": tools_planned
    }


async def run_react_workflow(
    user_request: str,
    planner,
    executor_client,
    database: Optional[RetailDatabase] = None,
    reasoning_effort: str = None,
    verbose: bool = True,
    use_fewshot: bool = True
) -> Dict[str, Any]:
    """
    Execute the ReAct workflow with Planner + ExecutorAgent (optimized).

    This workflow:
    1. Gets a plan from the Planner (predicts tools needed)
    2. Creates an ExecutorAgent with ONLY the predicted tools
    3. Executes the task using ReAct pattern

    Benefits:
    - Reduced cognitive load on executor (fewer tools to consider)
    - Potentially faster and more accurate responses
    - Token savings (~1200 tokens per request on average)

    The ExecutorAgent uses Agent Framework's native ReAct loop:
    - Single executor.run() call
    - Agent iterates: tool call → observe → decide → repeat
    - Continues until task is complete

    Args:
        user_request: Customer's natural language request
        planner: Planner agent (fine-tuned model)
        executor_client: Client for creating executor agents
        database: RetailDatabase instance (loads default if None)
        reasoning_effort: Reasoning effort for executor ("none", "low", "medium", "high")
        verbose: Print progress
        use_fewshot: Include few-shot example in executor prompt. Defaults to True.

    Returns:
        Dict with plan, execution results:
        - user_request: Original request
        - success: Whether workflow completed
        - plan_text: Planner's output
        - tools_planned: Tools predicted by Planner
        - tools_passed: Number of tools passed to executor
        - tools_executed: Tools actually executed
        - final_response: Customer-facing response
        - execution_trace: Detailed tool call trace
        - iterations: Number of tool calls
        - execution_time_s: Total execution time
    """
    from src.multiagent.agents import create_executor_with_tools

    # Initialize
    reset_tool_log()
    start_time = time.time()

    if database is None:
        database = load_database()

    # Use snapshot for isolation (each workflow gets its own copy)
    db_snapshot = database.snapshot()
    set_active_database(db_snapshot)

    results = {
        "user_request": user_request,
        "success": True,
        "plan_text": "",
        "tools_planned": [],
        "tools_passed": 0,
        "tools_executed": [],
        "execution_trace": [],
        "tool_calls": [],
        "final_response": "",
        "database_mutations": [],
        "iterations": 0,
        "execution_time_s": 0.0
    }

    try:
        # =====================================================================
        # Step 1: PLANNER generates the plan
        # =====================================================================
        if verbose:
            print("\n[1] PLANNER")

        # Fresh thread for Planner to avoid context contamination
        planner_thread = planner.get_new_thread()
        plan_result = await planner.run(user_request, thread=planner_thread)

        # Handle both structured output (Pydantic) and text output
        if hasattr(plan_result, 'value') and plan_result.value is not None:
            results["tools_planned"] = extract_tools_from_plan(plan_result.value)
            results["plan_text"] = json.dumps({"tools": results["tools_planned"]})
        else:
            results["plan_text"] = plan_result.text
            results["tools_planned"] = extract_tools_from_plan(plan_result.text)

        results["tools_passed"] = len(results["tools_planned"])

        if verbose:
            print(f"   Tools planned: {results['tools_planned']}")

        # =====================================================================
        # Step 2: EXECUTOR with filtered tools
        # =====================================================================
        if verbose:
            print(f"\n[2] EXECUTOR ({results['tools_passed']} tools)")

        # Create executor with only predicted tools and dynamic prompt
        # Tool descriptions come via Agent Framework API (from docstrings)
        # Additional context comes from prompt (from tool_definitions.json)
        executor = create_executor_with_tools(
            executor_client,
            tool_names=results["tools_planned"],
            reasoning_effort=reasoning_effort,
            dynamic_prompt=True,
            use_fewshot=use_fewshot
        )

        # Task with Planner guidance - executor knows which tools to use
        task = f"""Customer request: {user_request}

The Planner has determined these tools are needed: {results['tools_planned']}

Execute ALL the necessary tools to fulfill this request completely.
If the user has multiple orders, check each one to find the right one (e.g., find the pending order).
Make sure to complete the final action (exchange, return, cancel, etc.) - don't stop at just gathering information.
Generate a professional customer service response when complete."""

        # Fresh thread for isolation
        thread = executor.get_new_thread()

        # Single call - agent iterates automatically via ReAct pattern
        executor_result = await executor.run(task, thread=thread)

        # Extract final response
        results["final_response"] = executor_result.text

        # =====================================================================
        # Step 3: Collect trace data
        # =====================================================================
        tool_log = get_tool_log()
        results["tool_calls"] = tool_log
        results["tools_executed"] = list(set(tc["tool"] for tc in tool_log))
        results["iterations"] = len(tool_log)
        results["database_mutations"] = db_snapshot.get_mutations()

        results["execution_trace"] = [
            {
                "tool": tc["tool"],
                "agent": "executor",
                "success": "Error" not in str(tc.get("result", "")),
                "arguments": tc["arguments"],
                "result": tc["result"]
            }
            for tc in tool_log
        ]

        if verbose:
            print(f"   Executed: {results['tools_executed']}")
            print(f"   Iterations: {results['iterations']}")
            for tc in tool_log:
                status = "✓" if "Error" not in str(tc.get("result", "")) else "✗"
                print(f"   {status} {tc['tool']}({tc['arguments']})")
            if results["database_mutations"]:
                print(f"   Mutations: {len(results['database_mutations'])}")
            print(f"\n[3] RESPONSE")
            print(f"   {results['final_response'][:200]}...")

    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        if verbose:
            print(f"\n[ERROR] {e}")

    results["execution_time_s"] = round(time.time() - start_time, 2)

    return results
