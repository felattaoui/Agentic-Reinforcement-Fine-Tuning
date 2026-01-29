"""
Agent creation utilities for the multi-agent workflow.

Provides functions to create:
- ExecutorAgent (ReAct pattern with filtered tools)
- Planners (Baseline and Fine-tuned)
"""

from typing import Dict, Any, List

from src.settings import load_executor_prompt
from src.multiagent.tools import get_all_tools, get_tools_by_names


def create_executor_with_tools(
    client,
    tool_names: List[str] = None,
    reasoning_effort: str = None,
    dynamic_prompt: bool = True,
    use_fewshot: bool = True
) -> Any:
    """
    Create ExecutorAgent with specific tools (or all tools if None).

    This function enables dynamic tool filtering based on Planner predictions.
    Pass only the tools predicted by the Planner to reduce cognitive load
    on the executor model and potentially improve accuracy.

    Args:
        client: OpenAIResponsesClient for the executor model
        tool_names: List of tool names to include. If None or empty, uses all 15 tools.
        reasoning_effort: Optional reasoning effort ("none", "low", "medium", "high")
        dynamic_prompt: If True, uses a prompt with only relevant tool context
                       (without the static 15-tool list). Defaults to True.
        use_fewshot: If True, includes a few-shot example matching the action type.
                    Only applies when dynamic_prompt=True. Defaults to True.

    Returns:
        ExecutorAgent with specified tools attached

    Example:
        >>> # Create executor with only predicted tools (dynamic prompt + fewshot)
        >>> executor = create_executor_with_tools(client, ["find_user_id_by_email", "get_order_details"])
        >>>
        >>> # Create executor without fewshot (for A/B testing)
        >>> executor = create_executor_with_tools(client, tool_names, use_fewshot=False)
        >>>
        >>> # Create executor with all tools (static prompt)
        >>> executor_all = create_executor_with_tools(client, tool_names=None, dynamic_prompt=False)
    """
    if tool_names:
        tools = get_tools_by_names(tool_names)
    else:
        tools = get_all_tools()

    # Fallback to all tools if none of the requested tools were found
    if not tools:
        tools = get_all_tools()
        tool_names = None  # Reset to use static prompt

    # Choose prompt based on dynamic_prompt flag
    if dynamic_prompt and tool_names:
        from src.settings import load_executor_prompt_dynamic
        instructions = load_executor_prompt_dynamic(tool_names, use_fewshot=use_fewshot)
    else:
        instructions = load_executor_prompt()

    config = {
        "name": "ExecutorAgent",
        "instructions": instructions,
        "tools": tools
    }

    if reasoning_effort is not None:
        config["reasoning"] = {"effort": reasoning_effort}

    return client.create_agent(**config)


def create_planner(
    client,
    system_prompt: str,
    reasoning_effort: str = None,
    response_format: dict = None
) -> Any:
    """
    Create a single planner agent.

    Args:
        client: AzureOpenAIChatClient for the planner
        system_prompt: The planner system prompt
        reasoning_effort: Optional reasoning effort ("none", "low", "medium", "high")
        response_format: Optional JSON schema for structured output.
                        Use load_planner_schema() from settings to get the schema.

    Returns:
        Planner agent

    Example:
        >>> from src.settings import load_planner_schema
        >>> schema = load_planner_schema()
        >>> planner = create_planner(client, prompt, response_format=schema)
    """
    agent_config = {
        "name": "Planner",
        "instructions": system_prompt
    }

    if reasoning_effort is not None:
        agent_config["reasoning"] = {"effort": reasoning_effort}

    if response_format is not None:
        agent_config["response_format"] = response_format

    return client.create_agent(**agent_config)


def create_planners(baseline_client, finetuned_client, system_prompt: str) -> Dict[str, Any]:
    """
    Create baseline and fine-tuned planner agents.

    Args:
        baseline_client: AzureOpenAIChatClient for baseline (o4-mini)
        finetuned_client: AzureOpenAIChatClient for fine-tuned model
        system_prompt: The planner system prompt

    Returns:
        Dict with "baseline" and "finetuned" planners
    """
    baseline_planner = create_planner(baseline_client, system_prompt)
    finetuned_planner = create_planner(finetuned_client, system_prompt)

    return {
        "baseline": baseline_planner,
        "finetuned": finetuned_planner
    }
