"""
Utility tools adapted from tau-bench.

2 tools:
- transfer_to_human_agents
- calculate
"""

from typing import Annotated
from pydantic import Field
from agent_framework import ai_function

from .base import log_tool_call


@ai_function
def transfer_to_human_agents(
    summary: Annotated[str, Field(description="A summary of the user's issue.")]
) -> str:
    """Transfer the user to a human agent, with a summary of the user's issue. Only transfer if the user explicitly asks for a human agent, or if the user's issue cannot be resolved by the agent."""
    result = "Transfer successful"
    log_tool_call("transfer_to_human_agents", {"summary": summary}, result)
    return result


@ai_function
def calculate(
    expression: Annotated[str, Field(description="The mathematical expression to calculate, e.g., '(2+3)*4'.")]
) -> str:
    """Calculate the result of a mathematical expression. Only numbers, +, -, *, /, (, ), and spaces are allowed."""
    # Validate expression contains only allowed characters
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        result = "Error: expression contains invalid characters. Only numbers, +, -, *, /, (, ), and spaces are allowed."
        log_tool_call("calculate", {"expression": expression}, result)
        return result

    try:
        # Safe eval with restricted builtins
        calc_result = eval(expression, {"__builtins__": {}}, {})
        result = str(round(calc_result, 2))
        log_tool_call("calculate", {"expression": expression}, result)
        return result
    except Exception as e:
        result = f"Error: failed to calculate expression - {str(e)}"
        log_tool_call("calculate", {"expression": expression}, result)
        return result


def get_utility_tools() -> list:
    """Return all utility tools."""
    return [
        transfer_to_human_agents,
        calculate
    ]
