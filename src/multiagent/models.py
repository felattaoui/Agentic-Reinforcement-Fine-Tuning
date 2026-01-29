"""Pydantic models for structured outputs in multi-agent system."""

import json
from typing import List
from pydantic import BaseModel, field_validator


class SubAgentResponse(BaseModel):
    """Structured output from executor sub-agent.

    This model captures the result of a tool execution by a sub-agent,
    enabling programmatic decision-making for retry/replan logic.

    Attributes:
        tool_name: Name of the tool that was executed
        arguments_json: Arguments as JSON string (Azure strict mode requires fixed schemas)
        success: Whether the tool execution succeeded
        result: The result of the tool execution (if successful)
        error: Error message (if execution failed)
        needs_retry: Whether the operation should be retried

    Note:
        Azure OpenAI structured output requires `additionalProperties: false` for all
        object schemas. Since tool arguments vary by tool, we serialize them as a
        JSON string instead of using dict[str, Any].

    Example:
        >>> response = SubAgentResponse(
        ...     tool_name="get_order_details",
        ...     arguments_json='{"order_id": "12345"}',
        ...     success=True,
        ...     result="Order found: ..."
        ... )
        >>> response.get_arguments()
        {'order_id': '12345'}
    """
    tool_name: str
    arguments_json: str = "{}"  # JSON string instead of dict[str, Any]
    success: bool
    result: str | None = None
    error: str | None = None
    needs_retry: bool = False

    def get_arguments(self) -> dict:
        """Parse arguments_json back to a dictionary."""
        try:
            return json.loads(self.arguments_json)
        except json.JSONDecodeError:
            return {}

    @field_validator('arguments_json')
    @classmethod
    def validate_json(cls, v: str) -> str:
        """Ensure arguments_json is valid JSON."""
        if v:
            try:
                json.loads(v)
            except json.JSONDecodeError:
                # If invalid, return empty object
                return "{}"
        return v


class PlannerResponse(BaseModel):
    """Structured output from Planner agent.

    This model captures the Planner's decision about which tools to use
    for a given customer request.

    Attributes:
        tools: Ordered list of tool names to execute

    Note:
        This model aligns with src/config/planner_schema.json used in RFT training.
        The schema was simplified to only require 'tools' for more reliable parsing.

    Example:
        >>> response = PlannerResponse(
        ...     tools=["find_user_id_by_name_zip", "get_order_details", "cancel_pending_order"]
        ... )
    """
    tools: List[str]
