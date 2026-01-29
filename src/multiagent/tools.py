"""
Simulated tool functions for the multi-agent retail workflow.

These functions simulate the tau-bench retail API. They:
1. Receive real arguments extracted by sub-agents
2. Return plausible fake results for demonstration

This allows verification that:
- The Planner selected the right tools
- Sub-agents extracted correct arguments
- The workflow executes end-to-end
"""

from typing import Annotated, List, Dict, Any
from pydantic import Field
from agent_framework import ai_function


# =============================================================================
# TOOL CALL LOGGING
# =============================================================================

TOOL_CALL_LOG: List[Dict[str, Any]] = []


def log_tool_call(name: str, args: dict, result: dict) -> None:
    """Log a tool call for later analysis."""
    TOOL_CALL_LOG.append({
        "tool": name,
        "arguments": args,
        "result": result
    })


def reset_tool_log() -> None:
    """Reset the tool call log."""
    global TOOL_CALL_LOG
    TOOL_CALL_LOG = []


def get_tool_log() -> List[Dict[str, Any]]:
    """Get a copy of the current tool call log."""
    return TOOL_CALL_LOG.copy()


# =============================================================================
# ACCOUNT TOOLS (4)
# =============================================================================

@ai_function
def find_user_id_by_email(
    email: Annotated[str, Field(description="Customer email address")]
) -> dict:
    """Find user ID by their email address."""
    result = {
        "status": "success",
        "user_id": f"user_{hash(email) % 10000}",
        "email": email
    }
    log_tool_call("find_user_id_by_email", {"email": email}, result)
    return result


@ai_function
def find_user_id_by_name_zip(
    name: Annotated[str, Field(description="Customer full name")],
    zip_code: Annotated[str, Field(description="Customer zip code")]
) -> dict:
    """Find user ID by name and zip code."""
    result = {
        "status": "success",
        "user_id": f"user_{hash(name + zip_code) % 10000}",
        "name": name,
        "zip_code": zip_code
    }
    log_tool_call("find_user_id_by_name_zip", {"name": name, "zip_code": zip_code}, result)
    return result


@ai_function
def get_user_details(
    user_id: Annotated[str, Field(description="User ID")]
) -> dict:
    """Get detailed information about a user."""
    result = {
        "status": "success",
        "user_id": user_id,
        "name": "John Doe",
        "email": "john.doe@example.com",
        "address": "123 Main St, Anytown, USA 12345"
    }
    log_tool_call("get_user_details", {"user_id": user_id}, result)
    return result


@ai_function
def modify_user_address(
    user_id: Annotated[str, Field(description="User ID")],
    new_address: Annotated[str, Field(description="New address")]
) -> dict:
    """Modify a user's address."""
    result = {
        "status": "success",
        "user_id": user_id,
        "new_address": new_address,
        "message": "Address updated successfully"
    }
    log_tool_call("modify_user_address", {"user_id": user_id, "new_address": new_address}, result)
    return result


# =============================================================================
# ORDER TOOLS (7)
# =============================================================================

@ai_function
def get_order_details(
    order_id: Annotated[str, Field(description="Order ID (e.g., W123456)")]
) -> dict:
    """Get details of an order."""
    result = {
        "status": "success",
        "order_id": order_id,
        "order_status": "pending",
        "items": [{"name": "Running Shoes", "price": 99.99, "quantity": 1}],
        "total": 99.99,
        "payment_method": "credit_card"
    }
    log_tool_call("get_order_details", {"order_id": order_id}, result)
    return result


@ai_function
def cancel_pending_order(
    order_id: Annotated[str, Field(description="Order ID to cancel")],
    reason: Annotated[str, Field(description="Reason for cancellation")] = "customer_request"
) -> dict:
    """Cancel a pending order."""
    result = {
        "status": "success",
        "order_id": order_id,
        "cancellation_status": "cancelled",
        "reason": reason,
        "refund_amount": 99.99,
        "refund_timeline": "5-7 business days"
    }
    log_tool_call("cancel_pending_order", {"order_id": order_id, "reason": reason}, result)
    return result


@ai_function
def modify_pending_order_address(
    order_id: Annotated[str, Field(description="Order ID")],
    new_address: Annotated[str, Field(description="New shipping address")]
) -> dict:
    """Modify the shipping address of a pending order."""
    result = {
        "status": "success",
        "order_id": order_id,
        "new_address": new_address,
        "message": "Shipping address updated"
    }
    log_tool_call("modify_pending_order_address", {"order_id": order_id, "new_address": new_address}, result)
    return result


@ai_function
def modify_pending_order_items(
    order_id: Annotated[str, Field(description="Order ID")],
    new_item_ids: Annotated[str, Field(description="New item IDs")]
) -> dict:
    """Modify items in a pending order."""
    result = {
        "status": "success",
        "order_id": order_id,
        "new_item_ids": new_item_ids,
        "message": "Order items updated"
    }
    log_tool_call("modify_pending_order_items", {"order_id": order_id, "new_item_ids": new_item_ids}, result)
    return result


@ai_function
def modify_pending_order_payment(
    order_id: Annotated[str, Field(description="Order ID")],
    new_payment_method: Annotated[str, Field(description="New payment method")]
) -> dict:
    """Modify the payment method of a pending order."""
    result = {
        "status": "success",
        "order_id": order_id,
        "new_payment_method": new_payment_method,
        "message": "Payment method updated"
    }
    log_tool_call("modify_pending_order_payment", {"order_id": order_id, "new_payment_method": new_payment_method}, result)
    return result


@ai_function
def get_product_details(
    product_id: Annotated[str, Field(description="Product ID")]
) -> dict:
    """Get product details."""
    result = {
        "status": "success",
        "product_id": product_id,
        "name": "Running Shoes",
        "price": 99.99,
        "available_sizes": ["8", "9", "10", "11"],
        "in_stock": True
    }
    log_tool_call("get_product_details", {"product_id": product_id}, result)
    return result


@ai_function
def list_all_product_types() -> dict:
    """List all available product types."""
    result = {
        "status": "success",
        "product_types": [
            "Shoes", "Clothing", "Electronics", "Home & Garden",
            "Sports", "Books", "Toys", "Beauty"
        ]
    }
    log_tool_call("list_all_product_types", {}, result)
    return result


# =============================================================================
# REFUND TOOLS (2)
# =============================================================================

@ai_function
def return_delivered_order_items(
    order_id: Annotated[str, Field(description="Order ID")],
    item_ids: Annotated[str, Field(description="Item IDs to return")]
) -> dict:
    """Initiate return for delivered order items."""
    result = {
        "status": "success",
        "order_id": order_id,
        "item_ids": item_ids,
        "return_id": "RET_001",
        "return_label_url": "https://shipping.example.com/label/RET_001",
        "refund_amount": 99.99
    }
    log_tool_call("return_delivered_order_items", {"order_id": order_id, "item_ids": item_ids}, result)
    return result


@ai_function
def exchange_delivered_order_items(
    order_id: Annotated[str, Field(description="Order ID")],
    item_ids: Annotated[str, Field(description="Item IDs to exchange")],
    new_item_ids: Annotated[str, Field(description="New item IDs")]
) -> dict:
    """Exchange delivered order items."""
    result = {
        "status": "success",
        "order_id": order_id,
        "exchanged_items": item_ids,
        "new_items": new_item_ids,
        "exchange_id": "EXC_001",
        "shipping_estimate": "3-5 business days"
    }
    log_tool_call("exchange_delivered_order_items", {
        "order_id": order_id, "item_ids": item_ids, "new_item_ids": new_item_ids
    }, result)
    return result


# =============================================================================
# UTILITY TOOLS (2)
# =============================================================================

@ai_function
def transfer_to_human_agents(
    summary: Annotated[str, Field(description="Summary of the issue")]
) -> dict:
    """Transfer to human agent."""
    result = {
        "status": "success",
        "message": "Transferred to human agent",
        "queue_position": 3,
        "estimated_wait": "5 minutes"
    }
    log_tool_call("transfer_to_human_agents", {"summary": summary}, result)
    return result


@ai_function
def calculate(
    expression: Annotated[str, Field(description="Mathematical expression to calculate")]
) -> dict:
    """Perform a calculation."""
    try:
        # Safe eval for simple math
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            result_value = eval(expression)
        else:
            result_value = "Invalid expression"
    except:
        result_value = "Error in calculation"
    
    result = {
        "status": "success",
        "expression": expression,
        "result": result_value
    }
    log_tool_call("calculate", {"expression": expression}, result)
    return result


# =============================================================================
# TOOL COLLECTIONS
# =============================================================================

def get_account_tools() -> list:
    """Get account-related tools."""
    return [find_user_id_by_email, find_user_id_by_name_zip, get_user_details, modify_user_address]


def get_order_tools() -> list:
    """Get order-related tools."""
    return [
        get_order_details, cancel_pending_order, modify_pending_order_address,
        modify_pending_order_items, modify_pending_order_payment,
        get_product_details, list_all_product_types
    ]


def get_refund_tools() -> list:
    """Get refund-related tools."""
    return [return_delivered_order_items, exchange_delivered_order_items]


def get_utility_tools() -> list:
    """Get utility tools."""
    return [transfer_to_human_agents, calculate]


def get_all_tools() -> list:
    """Get all 15 tools."""
    return get_account_tools() + get_order_tools() + get_refund_tools() + get_utility_tools()


def print_tools_summary() -> None:
    """Print a summary of available tools."""
    print("✅ Simulated tool functions defined (15 tools)")
    print("   Account (4): find_user_id_by_email, find_user_id_by_name_zip, get_user_details, modify_user_address")
    print("   Order (7): get_order_details, cancel_pending_order, modify_pending_order_*, get_product_details, list_all_product_types")
    print("   Refund (2): return_delivered_order_items, exchange_delivered_order_items")
    print("   Utility (2): transfer_to_human_agents, calculate")
