"""
Retail tools adapted from tau-bench for agent_framework.

15 tools total:
- Account (4): find_user_id_by_email, find_user_id_by_name_zip, get_user_details, modify_user_address
- Order (7): get_order_details, cancel_pending_order, modify_pending_order_*, get_product_details, list_all_product_types
- Refund (2): return_delivered_order_items, exchange_delivered_order_items
- Utility (2): transfer_to_human_agents, calculate
"""

from .base import (
    set_active_database,
    get_active_database,
    clear_active_database,
    log_tool_call,
    reset_tool_log,
    get_tool_log
)

from .account import (
    find_user_id_by_email,
    find_user_id_by_name_zip,
    get_user_details,
    modify_user_address,
    get_account_tools
)

from .order import (
    get_order_details,
    cancel_pending_order,
    modify_pending_order_address,
    modify_pending_order_items,
    modify_pending_order_payment,
    get_product_details,
    list_all_product_types,
    get_order_tools
)

from .refund import (
    return_delivered_order_items,
    exchange_delivered_order_items,
    get_refund_tools
)

from .utility import (
    transfer_to_human_agents,
    calculate,
    get_utility_tools
)


def get_all_tools() -> list:
    """Get all 15 retail tools."""
    return get_account_tools() + get_order_tools() + get_refund_tools() + get_utility_tools()


# List of all known tool names for parsing
KNOWN_TOOLS = [
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
    "modify_user_address",
    "get_order_details",
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "get_product_details",
    "list_all_product_types",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "transfer_to_human_agents",
    "calculate"
]


# Tool name -> function registry for dynamic filtering
TOOL_REGISTRY = {
    "find_user_id_by_email": find_user_id_by_email,
    "find_user_id_by_name_zip": find_user_id_by_name_zip,
    "get_user_details": get_user_details,
    "modify_user_address": modify_user_address,
    "get_order_details": get_order_details,
    "cancel_pending_order": cancel_pending_order,
    "modify_pending_order_address": modify_pending_order_address,
    "modify_pending_order_items": modify_pending_order_items,
    "modify_pending_order_payment": modify_pending_order_payment,
    "get_product_details": get_product_details,
    "list_all_product_types": list_all_product_types,
    "return_delivered_order_items": return_delivered_order_items,
    "exchange_delivered_order_items": exchange_delivered_order_items,
    "transfer_to_human_agents": transfer_to_human_agents,
    "calculate": calculate
}


def get_tools_by_names(tool_names: list) -> list:
    """
    Get tool functions by their string names.

    Args:
        tool_names: List of tool name strings (e.g., ["find_user_id_by_email"])

    Returns:
        List of tool functions for Agent Framework
    """
    tools = []
    for name in tool_names:
        if name in TOOL_REGISTRY:
            tools.append(TOOL_REGISTRY[name])
    return tools


# Tool descriptions extracted from docstrings
TOOL_DESCRIPTIONS = {
    "find_user_id_by_email": "Find user id by email",
    "find_user_id_by_name_zip": "Find user id by first name, last name, and zip code",
    "get_user_details": "Get the details of a user, including their orders",
    "modify_user_address": "Modify the default address of a user",
    "get_order_details": "Get the details of an order",
    "cancel_pending_order": "Cancel a pending order. If the order is not pending, it cannot be cancelled",
    "modify_pending_order_address": "Modify the shipping address of a pending order",
    "modify_pending_order_items": "Modify items in a pending order (add/remove/change)",
    "modify_pending_order_payment": "Modify the payment method of a pending order",
    "get_product_details": "Get the details of a product",
    "list_all_product_types": "List all product types available in the catalog",
    "return_delivered_order_items": "Return items from a delivered order for refund",
    "exchange_delivered_order_items": "Exchange items from a delivered order for different items",
    "transfer_to_human_agents": "Transfer the conversation to a human agent",
    "calculate": "Calculate the result of a mathematical expression"
}


def get_tool_descriptions(tool_names: list) -> str:
    """
    Get formatted tool descriptions for a list of tool names.

    Args:
        tool_names: List of tool name strings

    Returns:
        Formatted string with tool names and descriptions
    """
    descriptions = []
    for name in tool_names:
        if name in TOOL_DESCRIPTIONS:
            descriptions.append(f"- {name}: {TOOL_DESCRIPTIONS[name]}")
    return "\n".join(descriptions) if descriptions else "No tools available"


def print_tools_summary() -> None:
    """Print a summary of available tools."""
    print("Available retail tools (15 total):")
    print("  Account (4): find_user_id_by_email, find_user_id_by_name_zip, get_user_details, modify_user_address")
    print("  Order (7): get_order_details, cancel_pending_order, modify_pending_order_*, get_product_details, list_all_product_types")
    print("  Refund (2): return_delivered_order_items, exchange_delivered_order_items")
    print("  Utility (2): transfer_to_human_agents, calculate")


__all__ = [
    # Base utilities
    "set_active_database",
    "get_active_database",
    "clear_active_database",
    "log_tool_call",
    "reset_tool_log",
    "get_tool_log",
    # Account tools
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
    "modify_user_address",
    "get_account_tools",
    # Order tools
    "get_order_details",
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "get_product_details",
    "list_all_product_types",
    "get_order_tools",
    # Refund tools
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "get_refund_tools",
    # Utility tools
    "transfer_to_human_agents",
    "calculate",
    "get_utility_tools",
    # All tools
    "get_all_tools",
    "KNOWN_TOOLS",
    "TOOL_REGISTRY",
    "TOOL_DESCRIPTIONS",
    "get_tools_by_names",
    "get_tool_descriptions",
    "print_tools_summary"
]
