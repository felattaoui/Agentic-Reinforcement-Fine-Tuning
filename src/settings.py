"""
Centralized configuration for the RFT workshop.

This module provides:
- Path definitions (DATA_DIR, CONFIG_DIR, etc.)
- Environment variable loading
- Azure OpenAI configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

def get_root_dir() -> Path:
    """
    Get the project root directory.
    Works whether called from notebooks/ or from root.
    """
    # Try to find root by looking for .env or README.md
    current = Path(__file__).parent.parent  # src/ -> root
    if (current / ".env").exists() or (current / "README.md").exists():
        return current
    
    # Fallback: assume we're in notebooks/
    return Path("..").resolve()


ROOT_DIR = get_root_dir()
DATA_DIR = ROOT_DIR / "data"
CONFIG_DIR = ROOT_DIR / "src" / "config"
GRADERS_DIR = ROOT_DIR / "src" / "graders"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Subdirectories for data (inputs)
DATASETS_DIR = DATA_DIR / "datasets"

# Subdirectories for outputs (generated artifacts)
PLANNER_OUTPUTS_DIR = OUTPUTS_DIR / "planner"
PLANNER_RESPONSES_DIR = PLANNER_OUTPUTS_DIR / "responses"
MULTIAGENT_OUTPUTS_DIR = OUTPUTS_DIR / "multiagent"
EDA_OUTPUTS_DIR = OUTPUTS_DIR / "eda"

# Ensure directories exist
for directory in [
    DATA_DIR, CONFIG_DIR, GRADERS_DIR, OUTPUTS_DIR,
    DATASETS_DIR,
    PLANNER_OUTPUTS_DIR, PLANNER_RESPONSES_DIR,
    MULTIAGENT_OUTPUTS_DIR, EDA_OUTPUTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

# Load .env from root
load_dotenv(ROOT_DIR / ".env")

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o4-mini")
AZURE_DEPLOYMENT_DATA = os.getenv("AZURE_OPENAI_DEPLOYMENT_DATA", "gpt-5.1")
AZURE_DEPLOYMENT_VANILLA = os.getenv("AZURE_OPENAI_DEPLOYMENT_VANILLA", "gpt-5.2")
FINETUNED_DEPLOYMENT = os.getenv("FINETUNED_DEPLOYMENT", "retail-agent-ft")
SUBAGENT_DEPLOYMENT = os.getenv("SUBAGENT_DEPLOYMENT", "gpt-5-mini")
EVAL_DEPLOYMENT = os.getenv("EVAL_DEPLOYMENT", "gpt-4.1-mini")
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Derived values
RESOURCE_NAME = AZURE_ENDPOINT.split("//")[1].split(".")[0] if AZURE_ENDPOINT else None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_config():
    """Print current configuration for debugging."""
    print(f"📁 Root dir: {ROOT_DIR.absolute()}")
    print(f"📁 Data dir: {DATA_DIR.absolute()}")
    print(f"📁 Config dir: {CONFIG_DIR.absolute()}")
    print(f"📁 Graders dir: {GRADERS_DIR.absolute()}")
    print(f"📁 Outputs dir: {OUTPUTS_DIR.absolute()}")
    print(f"🔗 Endpoint: {AZURE_ENDPOINT}")
    print(f"📦 Baseline deployment: {AZURE_DEPLOYMENT}")
    print(f"🆕 Vanilla deployment: {AZURE_DEPLOYMENT_VANILLA}")
    print(f"🎯 Fine-tuned deployment: {FINETUNED_DEPLOYMENT}")


def load_system_prompt() -> str:
    """
    Load the planner system prompt from training data (single source of truth).

    The prompt is extracted from the first sample in train.jsonl to ensure
    consistency between training and inference. This avoids duplication and
    ensures the model is always evaluated with the same prompt it was trained on.

    Returns:
        The system prompt string used for training
    """
    import json
    train_file = DATASETS_DIR / "train.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found at {train_file}")

    with open(train_file, encoding="utf-8") as f:
        first_sample = json.loads(f.readline())

    # The system prompt is the first message (role: "developer" or "system")
    return first_sample["messages"][0]["content"]


def load_tool_definitions() -> list:
    """Load tool definitions from config directory."""
    import json
    tools_path = CONFIG_DIR / "tool_definitions.json"
    if not tools_path.exists():
        raise FileNotFoundError(f"Tool definitions not found at {tools_path}")
    with open(tools_path) as f:
        return json.load(f)


def load_planner_schema() -> dict:
    """
    Load the Planner structured output schema from config directory.

    This schema defines the expected JSON structure for Planner responses
    when using structured output mode. The schema enforces:
    - reasoning: Explanation of tool selection
    - tools: Ordered list of tools to execute (from the 15 available tools)

    Returns:
        Dict containing the JSON schema for response_format parameter

    Example:
        >>> schema = load_planner_schema()
        >>> planner = create_planner(client, prompt, response_format=schema)
    """
    import json
    schema_path = CONFIG_DIR / "planner_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Planner schema not found at {schema_path}")
    with open(schema_path) as f:
        return json.load(f)


def load_executor_prompt() -> str:
    """
    Load the ExecutorAgent system prompt (static, with all 15 tools).

    The ExecutorAgent uses the ReAct pattern (Reasoning + Acting) to
    iterate through tools until the task is complete. It receives all
    15 retail tools and decides which to call based on the Planner's
    suggestions and the observed results.

    Returns:
        The executor prompt string

    Example:
        >>> prompt = load_executor_prompt()
        >>> executor = client.create_agent(name="Executor", instructions=prompt, tools=get_all_tools())
    """
    prompt_path = CONFIG_DIR / "executor_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Executor prompt not found at {prompt_path}")
    return prompt_path.read_text()


def load_tool_context(tool_names: list) -> str:
    """
    Load additional context for specific tools from tool_definitions.json.

    Returns only the 'context' field for tools that have additional
    usage notes beyond their docstrings. Tools without context are skipped.

    Args:
        tool_names: List of tool names to get context for

    Returns:
        Formatted string with tool context notes, or empty string if none

    Example:
        >>> context = load_tool_context(["cancel_pending_order", "get_order_details"])
        >>> print(context)
        - cancel_pending_order: Ask for explicit user confirmation...
    """
    import json
    tools_path = CONFIG_DIR / "tool_definitions.json"
    if not tools_path.exists():
        return ""

    with open(tools_path) as f:
        all_tools = json.load(f)

    contexts = []
    for tool in all_tools:
        if tool["name"] in tool_names and tool.get("context"):
            contexts.append(f"- {tool['name']}: {tool['context']}")

    return "\n".join(contexts) if contexts else ""


def load_fewshot_examples() -> list:
    """
    Load few-shot examples from fewshot_examples.json.

    These examples were selected from train set with F2=1 and TCA>=4,
    covering different action types (exchange, cancel, return, modify).

    Returns:
        List of example dicts with action_type, trace, etc.
    """
    import json
    examples_path = CONFIG_DIR / "fewshot_examples.json"
    if not examples_path.exists():
        return []
    with open(examples_path) as f:
        return json.load(f)


def select_fewshot_example(tools_predicted: list) -> dict | None:
    """
    Select ONE few-shot example based on the action tool in tools_predicted.

    Maps action tools to action types and returns a matching example.

    Args:
        tools_predicted: List of tool names predicted by the Planner

    Returns:
        A matching example dict, or None if no action tool found

    Example:
        >>> example = select_fewshot_example(["find_user_id_by_email", "cancel_pending_order"])
        >>> example["action_type"]  # "cancel"
    """
    # Mapping: action tool -> action type
    action_tools = {
        "cancel_pending_order": "cancel",
        "return_delivered_order_items": "return",
        "exchange_delivered_order_items": "exchange",
        "modify_pending_order_items": "modify",
        "modify_pending_order_address": "modify",
        "modify_pending_order_payment": "modify",
    }

    # Find the action type from predicted tools
    action_type = None
    for tool in tools_predicted:
        if tool in action_tools:
            action_type = action_tools[tool]
            break

    if not action_type:
        return None

    # Load examples and return first matching
    examples = load_fewshot_examples()
    for ex in examples:
        if ex["action_type"] == action_type:
            return ex

    return None


def format_fewshot_example(example: dict) -> str:
    """
    Format a few-shot example for inclusion in the prompt.

    Args:
        example: Example dict from fewshot_examples.json

    Returns:
        Formatted string showing the workflow steps
    """
    if not example:
        return ""

    lines = [f"### Example ({example['action_type'].title()} workflow)"]
    lines.append(f"Customer: \"{example['query_summary'][:100]}...\"")
    lines.append("")

    for i, step in enumerate(example["trace"], 1):
        tool = step["tool"]
        args = step["args"]
        result = step["result_summary"][:50] if len(step["result_summary"]) > 50 else step["result_summary"]

        # Format args concisely
        args_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}'
                            for k, v in args.items())

        # Mark the final action tool
        is_action = tool in ["cancel_pending_order", "return_delivered_order_items",
                            "exchange_delivered_order_items", "modify_pending_order_items",
                            "modify_pending_order_address", "modify_pending_order_payment"]
        action_marker = "  <- ACTION" if is_action else ""

        lines.append(f"{i}. {tool}({args_str}) -> {result}{action_marker}")

    return "\n".join(lines)


def load_executor_prompt_dynamic(tool_names: list, use_fewshot: bool = True) -> str:
    """
    Load the ExecutorAgent prompt with dynamic tool context and optional few-shot example.

    This creates a prompt without the static 15-tool list, and instead:
    1. Appends only the relevant context for the tools being used
    2. Optionally includes ONE few-shot example matching the action type

    Args:
        tool_names: List of tool names the executor will have access to
        use_fewshot: If True, includes a few-shot example. Defaults to True.

    Returns:
        The executor prompt with dynamic content

    Example:
        >>> prompt = load_executor_prompt_dynamic(["find_user_id_by_name_zip", "cancel_pending_order"])
        >>> # Prompt will include a cancel example and context notes
        >>>
        >>> prompt_no_fewshot = load_executor_prompt_dynamic(tool_names, use_fewshot=False)
        >>> # Prompt with only context notes, no example
    """
    base_path = CONFIG_DIR / "executor_prompt_base.txt"
    if not base_path.exists():
        raise FileNotFoundError(f"Executor base prompt not found at {base_path}")

    base_prompt = base_path.read_text()
    sections = []

    # Add few-shot example based on action type (if enabled)
    if use_fewshot:
        example = select_fewshot_example(tool_names)
        if example:
            example_section = f"""## Example

{format_fewshot_example(example)}

**Important:** Notice how the workflow ends with the ACTION tool. Do NOT stop before executing the action.
"""
            sections.append(example_section)

    # Add tool context notes
    tool_context = load_tool_context(tool_names)
    if tool_context:
        context_section = f"""## Tool Usage Notes

{tool_context}
"""
        sections.append(context_section)

    if sections:
        return base_prompt + "\n" + "\n".join(sections)

    return base_prompt
