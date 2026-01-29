"""
Order tools adapted from tau-bench.

7 tools:
- get_order_details
- cancel_pending_order
- modify_pending_order_address
- modify_pending_order_items
- modify_pending_order_payment
- get_product_details
- list_all_product_types
"""

import json
from typing import Annotated, List
from pydantic import Field
from agent_framework import ai_function

from .base import get_active_database, log_tool_call


@ai_function
def get_order_details(
    order_id: Annotated[str, Field(description="The order id, such as '#W0000000'. Be careful to include the '#' symbol.")]
) -> str:
    """Get the status and details of an order."""
    db = get_active_database()
    orders = db.orders

    if order_id in orders:
        result = json.dumps(orders[order_id])
        log_tool_call("get_order_details", {"order_id": order_id}, result)
        return result

    result = "Error: order not found"
    log_tool_call("get_order_details", {"order_id": order_id}, result)
    return result


@ai_function
def cancel_pending_order(
    order_id: Annotated[str, Field(description="The order id, such as '#W0000000'.")],
    reason: Annotated[str, Field(description="The reason for cancellation. Must be 'no longer needed' or 'ordered by mistake'.")]
) -> str:
    """Cancel a pending order. Only pending orders can be cancelled."""
    db = get_active_database()
    orders = db.orders
    users = db.users

    if order_id not in orders:
        result = "Error: order not found"
        log_tool_call("cancel_pending_order", {"order_id": order_id, "reason": reason}, result)
        return result

    order = orders[order_id]

    if order["status"] != "pending":
        result = f"Error: order status is {order['status']}, only pending orders can be cancelled"
        log_tool_call("cancel_pending_order", {"order_id": order_id, "reason": reason}, result)
        return result

    if reason not in ["no longer needed", "ordered by mistake"]:
        result = "Error: reason must be 'no longer needed' or 'ordered by mistake'"
        log_tool_call("cancel_pending_order", {"order_id": order_id, "reason": reason}, result)
        return result

    # Process refunds
    user_id = order["user_id"]
    user = users.get(user_id, {})
    payment_methods = user.get("payment_methods", {})

    for payment in order.get("payment_history", []):
        if payment["transaction_type"] == "payment":
            refund = {
                "transaction_type": "refund",
                "amount": payment["amount"],
                "payment_method_id": payment["payment_method_id"]
            }
            order["payment_history"].append(refund)

            # Update gift card balance if applicable
            pm_id = payment["payment_method_id"]
            if "gift_card" in pm_id and pm_id in payment_methods:
                payment_methods[pm_id]["balance"] = round(
                    payment_methods[pm_id].get("balance", 0) + payment["amount"], 2
                )

    order["status"] = "cancelled"
    order["cancel_reason"] = reason

    result = json.dumps(order)
    log_tool_call("cancel_pending_order", {"order_id": order_id, "reason": reason}, result)
    db.record_mutation("cancel_pending_order", {"order_id": order_id, "reason": reason}, result)
    return result


@ai_function
def modify_pending_order_address(
    order_id: Annotated[str, Field(description="The order id, such as '#W0000000'.")],
    address1: Annotated[str, Field(description="The first line of the address.")],
    address2: Annotated[str, Field(description="The second line of the address.")],
    city: Annotated[str, Field(description="The city.")],
    state: Annotated[str, Field(description="The state.")],
    country: Annotated[str, Field(description="The country.")],
    zip: Annotated[str, Field(description="The zip code.")]
) -> str:
    """Modify the shipping address of a pending order."""
    db = get_active_database()
    orders = db.orders

    if order_id not in orders:
        result = "Error: order not found"
        log_tool_call("modify_pending_order_address", {"order_id": order_id}, result)
        return result

    order = orders[order_id]

    if order["status"] != "pending":
        result = f"Error: order status is {order['status']}, only pending orders can be modified"
        log_tool_call("modify_pending_order_address", {"order_id": order_id}, result)
        return result

    order["address"] = {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip
    }

    result = json.dumps(order)
    log_tool_call("modify_pending_order_address", {"order_id": order_id, "address1": address1, "address2": address2, "city": city, "state": state, "country": country, "zip": zip}, result)
    db.record_mutation("modify_pending_order_address", {"order_id": order_id}, result)
    return result


@ai_function
def modify_pending_order_items(
    order_id: Annotated[str, Field(description="The order id, such as '#W0000000'.")],
    item_ids: Annotated[List[str], Field(description="List of item ids to be modified, e.g., ['1000000', '1000001'].")],
    new_item_ids: Annotated[List[str], Field(description="List of new item ids, e.g., ['2000000', '2000001'].")],
    payment_method_id: Annotated[str, Field(description="The payment method id to use for price difference.")]
) -> str:
    """Modify items in a pending order. For each item, the new item must be of the same product type."""
    db = get_active_database()
    orders = db.orders
    users = db.users
    products = db.products

    if order_id not in orders:
        result = "Error: order not found"
        log_tool_call("modify_pending_order_items", {"order_id": order_id}, result)
        return result

    order = orders[order_id]

    if order["status"] != "pending":
        result = f"Error: order status is {order['status']}, only pending orders can be modified"
        log_tool_call("modify_pending_order_items", {"order_id": order_id}, result)
        return result

    # Item lists (already lists)
    item_id_list = item_ids
    new_item_id_list = new_item_ids

    if len(item_id_list) != len(new_item_id_list):
        result = "Error: item_ids and new_item_ids must have the same length"
        log_tool_call("modify_pending_order_items", {"order_id": order_id}, result)
        return result

    # Validate items exist in order
    order_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_id_list:
        if item_id not in order_item_ids:
            result = f"Error: item {item_id} not found in order"
            log_tool_call("modify_pending_order_items", {"order_id": order_id}, result)
            return result

    # Calculate price difference and update items
    price_diff = 0.0
    for old_id, new_id in zip(item_id_list, new_item_id_list):
        # Find old item
        old_item = next(item for item in order["items"] if item["item_id"] == old_id)
        old_price = old_item["price"]

        # Find new item in products
        new_item_found = False
        for product_id, product in products.items():
            if new_id in product.get("variants", {}):
                variant = product["variants"][new_id]
                if not variant.get("available", False):
                    result = f"Error: item {new_id} is not available"
                    log_tool_call("modify_pending_order_items", {"order_id": order_id}, result)
                    return result
                new_price = variant["price"]
                price_diff += new_price - old_price

                # Update item in order
                old_item["item_id"] = new_id
                old_item["price"] = new_price
                old_item["options"] = variant.get("options", {})
                new_item_found = True
                break

        if not new_item_found:
            result = f"Error: item {new_id} not found in products"
            log_tool_call("modify_pending_order_items", {"order_id": order_id}, result)
            return result

    # Handle payment for price difference
    user_id = order["user_id"]
    user = users.get(user_id, {})
    payment_methods = user.get("payment_methods", {})

    if payment_method_id not in payment_methods:
        result = f"Error: payment method {payment_method_id} not found"
        log_tool_call("modify_pending_order_items", {"order_id": order_id}, result)
        return result

    price_diff = round(price_diff, 2)

    if price_diff > 0:
        # Need to charge more
        if "gift_card" in payment_method_id:
            balance = payment_methods[payment_method_id].get("balance", 0)
            if balance < price_diff:
                result = f"Error: gift card balance ({balance}) insufficient for price difference ({price_diff})"
                log_tool_call("modify_pending_order_items", {"order_id": order_id}, result)
                return result
            payment_methods[payment_method_id]["balance"] = round(balance - price_diff, 2)

        order["payment_history"].append({
            "transaction_type": "payment",
            "amount": price_diff,
            "payment_method_id": payment_method_id
        })
    elif price_diff < 0:
        # Need to refund
        order["payment_history"].append({
            "transaction_type": "refund",
            "amount": abs(price_diff),
            "payment_method_id": payment_method_id
        })
        if "gift_card" in payment_method_id:
            payment_methods[payment_method_id]["balance"] = round(
                payment_methods[payment_method_id].get("balance", 0) + abs(price_diff), 2
            )

    result = json.dumps(order)
    log_tool_call("modify_pending_order_items", {"order_id": order_id, "item_ids": item_ids, "new_item_ids": new_item_ids, "payment_method_id": payment_method_id}, result)
    db.record_mutation("modify_pending_order_items", {"order_id": order_id}, result)
    return result


@ai_function
def modify_pending_order_payment(
    order_id: Annotated[str, Field(description="The order id, such as '#W0000000'.")],
    payment_method_id: Annotated[str, Field(description="The new payment method id to use.")]
) -> str:
    """Modify the payment method of a pending order."""
    db = get_active_database()
    orders = db.orders
    users = db.users

    if order_id not in orders:
        result = "Error: order not found"
        log_tool_call("modify_pending_order_payment", {"order_id": order_id}, result)
        return result

    order = orders[order_id]

    if order["status"] != "pending":
        result = f"Error: order status is {order['status']}, only pending orders can be modified"
        log_tool_call("modify_pending_order_payment", {"order_id": order_id}, result)
        return result

    user_id = order["user_id"]
    user = users.get(user_id, {})
    payment_methods = user.get("payment_methods", {})

    if payment_method_id not in payment_methods:
        result = f"Error: payment method {payment_method_id} not found for user"
        log_tool_call("modify_pending_order_payment", {"order_id": order_id}, result)
        return result

    # Get current payment
    payment_history = order.get("payment_history", [])
    if len(payment_history) != 1:
        result = "Error: order must have exactly one payment record"
        log_tool_call("modify_pending_order_payment", {"order_id": order_id}, result)
        return result

    current_payment = payment_history[0]
    if current_payment["payment_method_id"] == payment_method_id:
        result = "Error: new payment method is the same as current"
        log_tool_call("modify_pending_order_payment", {"order_id": order_id}, result)
        return result

    amount = current_payment["amount"]

    # Check gift card balance if new method is gift card
    if "gift_card" in payment_method_id:
        balance = payment_methods[payment_method_id].get("balance", 0)
        if balance < amount:
            result = f"Error: gift card balance ({balance}) insufficient for order ({amount})"
            log_tool_call("modify_pending_order_payment", {"order_id": order_id}, result)
            return result
        payment_methods[payment_method_id]["balance"] = round(balance - amount, 2)

    # Refund old payment method if gift card
    old_method = current_payment["payment_method_id"]
    if "gift_card" in old_method and old_method in payment_methods:
        payment_methods[old_method]["balance"] = round(
            payment_methods[old_method].get("balance", 0) + amount, 2
        )

    # Update payment history
    order["payment_history"].append({
        "transaction_type": "payment",
        "amount": amount,
        "payment_method_id": payment_method_id
    })
    order["payment_history"].append({
        "transaction_type": "refund",
        "amount": amount,
        "payment_method_id": old_method
    })

    result = json.dumps(order)
    log_tool_call("modify_pending_order_payment", {"order_id": order_id, "payment_method_id": payment_method_id}, result)
    db.record_mutation("modify_pending_order_payment", {"order_id": order_id}, result)
    return result


@ai_function
def get_product_details(
    product_id: Annotated[str, Field(description="The product id. Note: product id is different from item id.")]
) -> str:
    """Get the details of a product including all variants."""
    db = get_active_database()
    products = db.products

    if product_id in products:
        result = json.dumps(products[product_id])
        log_tool_call("get_product_details", {"product_id": product_id}, result)
        return result

    result = "Error: product not found"
    log_tool_call("get_product_details", {"product_id": product_id}, result)
    return result


@ai_function
def list_all_product_types() -> str:
    """List the name and product id of all product types. There are 50 product types in the store."""
    db = get_active_database()
    products = db.products

    product_list = [
        {"name": p["name"], "product_id": pid}
        for pid, p in products.items()
    ]
    product_list.sort(key=lambda x: x["name"])

    result = json.dumps(product_list)
    log_tool_call("list_all_product_types", {}, result)
    return result


def get_order_tools() -> list:
    """Return all order tools."""
    return [
        get_order_details,
        cancel_pending_order,
        modify_pending_order_address,
        modify_pending_order_items,
        modify_pending_order_payment,
        get_product_details,
        list_all_product_types
    ]
