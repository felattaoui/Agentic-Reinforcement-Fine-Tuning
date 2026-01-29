"""
Refund tools adapted from tau-bench.

2 tools:
- return_delivered_order_items
- exchange_delivered_order_items
"""

import json
from typing import Annotated, List
from pydantic import Field
from agent_framework import ai_function

from .base import get_active_database, log_tool_call


@ai_function
def return_delivered_order_items(
    order_id: Annotated[str, Field(description="The order id, such as '#W0000000'.")],
    item_ids: Annotated[List[str], Field(description="List of item ids to return, e.g., ['1000000', '1000001'].")],
    payment_method_id: Annotated[str, Field(description="The payment method id to receive the refund.")]
) -> str:
    """Return items from a delivered order. The order status will be changed to 'return requested'."""
    db = get_active_database()
    orders = db.orders
    users = db.users

    if order_id not in orders:
        result = "Error: order not found"
        log_tool_call("return_delivered_order_items", {"order_id": order_id}, result)
        return result

    order = orders[order_id]

    if order["status"] != "delivered":
        result = f"Error: order status is {order['status']}, only delivered orders can be returned"
        log_tool_call("return_delivered_order_items", {"order_id": order_id}, result)
        return result

    user_id = order["user_id"]
    user = users.get(user_id, {})
    payment_methods = user.get("payment_methods", {})

    # Validate payment method
    if payment_method_id not in payment_methods:
        # Check if it's the original payment method from order
        original_methods = [p["payment_method_id"] for p in order.get("payment_history", [])]
        if payment_method_id not in original_methods:
            result = f"Error: payment method {payment_method_id} not valid for refund"
            log_tool_call("return_delivered_order_items", {"order_id": order_id}, result)
            return result

    # Validate item ids (already a list)
    item_id_list = item_ids
    order_item_ids = [item["item_id"] for item in order["items"]]

    # Count occurrences for duplicate handling
    from collections import Counter
    requested_counts = Counter(item_id_list)
    available_counts = Counter(order_item_ids)

    for item_id, count in requested_counts.items():
        if item_id not in available_counts:
            result = f"Error: item {item_id} not found in order"
            log_tool_call("return_delivered_order_items", {"order_id": order_id}, result)
            return result
        if count > available_counts[item_id]:
            result = f"Error: requested {count} of item {item_id} but only {available_counts[item_id]} in order"
            log_tool_call("return_delivered_order_items", {"order_id": order_id}, result)
            return result

    # Calculate refund amount
    refund_amount = 0.0
    for item_id in item_id_list:
        item = next(item for item in order["items"] if item["item_id"] == item_id)
        refund_amount += item["price"]

    refund_amount = round(refund_amount, 2)

    # Update order status and add return details
    order["status"] = "return requested"
    order["return_items"] = item_id_list
    order["return_payment_method_id"] = payment_method_id
    order["refund_amount"] = refund_amount

    result = json.dumps(order)
    log_tool_call("return_delivered_order_items",
                  {"order_id": order_id, "item_ids": item_ids, "payment_method_id": payment_method_id},
                  result)
    db.record_mutation("return_delivered_order_items", {"order_id": order_id}, result)
    return result


@ai_function
def exchange_delivered_order_items(
    order_id: Annotated[str, Field(description="The order id, such as '#W0000000'.")],
    item_ids: Annotated[List[str], Field(description="List of item ids to exchange, e.g., ['1000000', '1000001'].")],
    new_item_ids: Annotated[List[str], Field(description="List of new item ids, e.g., ['2000000', '2000001'].")],
    payment_method_id: Annotated[str, Field(description="The payment method id for price difference.")]
) -> str:
    """Exchange items from a delivered order. The order status will be changed to 'exchange requested'."""
    db = get_active_database()
    orders = db.orders
    users = db.users
    products = db.products

    if order_id not in orders:
        result = "Error: order not found"
        log_tool_call("exchange_delivered_order_items", {"order_id": order_id}, result)
        return result

    order = orders[order_id]

    if order["status"] != "delivered":
        result = f"Error: order status is {order['status']}, only delivered orders can be exchanged"
        log_tool_call("exchange_delivered_order_items", {"order_id": order_id}, result)
        return result

    # Item lists (already lists)
    item_id_list = item_ids
    new_item_id_list = new_item_ids

    if len(item_id_list) != len(new_item_id_list):
        result = "Error: item_ids and new_item_ids must have the same length"
        log_tool_call("exchange_delivered_order_items", {"order_id": order_id}, result)
        return result

    # Validate items exist in order
    order_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_id_list:
        if item_id not in order_item_ids:
            result = f"Error: item {item_id} not found in order"
            log_tool_call("exchange_delivered_order_items", {"order_id": order_id}, result)
            return result

    user_id = order["user_id"]
    user = users.get(user_id, {})
    payment_methods = user.get("payment_methods", {})

    if payment_method_id not in payment_methods:
        result = f"Error: payment method {payment_method_id} not found"
        log_tool_call("exchange_delivered_order_items", {"order_id": order_id}, result)
        return result

    # Calculate price difference and validate new items
    price_diff = 0.0
    exchange_details = []

    for old_id, new_id in zip(item_id_list, new_item_id_list):
        # Find old item price
        old_item = next(item for item in order["items"] if item["item_id"] == old_id)
        old_price = old_item["price"]
        old_product_id = old_item["product_id"]

        # Find new item in products
        new_item_found = False
        for product_id, product in products.items():
            if new_id in product.get("variants", {}):
                # Validate same product type
                if product_id != old_product_id:
                    result = f"Error: new item {new_id} is not the same product type as {old_id}"
                    log_tool_call("exchange_delivered_order_items", {"order_id": order_id}, result)
                    return result

                variant = product["variants"][new_id]
                if not variant.get("available", False):
                    result = f"Error: item {new_id} is not available"
                    log_tool_call("exchange_delivered_order_items", {"order_id": order_id}, result)
                    return result

                new_price = variant["price"]
                price_diff += new_price - old_price
                exchange_details.append({
                    "old_item_id": old_id,
                    "new_item_id": new_id,
                    "old_price": old_price,
                    "new_price": new_price
                })
                new_item_found = True
                break

        if not new_item_found:
            result = f"Error: item {new_id} not found in products"
            log_tool_call("exchange_delivered_order_items", {"order_id": order_id}, result)
            return result

    price_diff = round(price_diff, 2)

    # Handle payment for price difference
    if price_diff > 0:
        if "gift_card" in payment_method_id:
            balance = payment_methods[payment_method_id].get("balance", 0)
            if balance < price_diff:
                result = f"Error: gift card balance ({balance}) insufficient for price difference ({price_diff})"
                log_tool_call("exchange_delivered_order_items", {"order_id": order_id}, result)
                return result

    # Update order status and add exchange details
    order["status"] = "exchange requested"
    order["exchange_items"] = item_id_list
    order["exchange_new_items"] = new_item_id_list
    order["exchange_payment_method_id"] = payment_method_id
    order["exchange_price_difference"] = price_diff

    result = json.dumps(order)
    log_tool_call("exchange_delivered_order_items",
                  {"order_id": order_id, "item_ids": item_ids, "new_item_ids": new_item_ids, "payment_method_id": payment_method_id},
                  result)
    db.record_mutation("exchange_delivered_order_items", {"order_id": order_id}, result)
    return result


def get_refund_tools() -> list:
    """Return all refund tools."""
    return [
        return_delivered_order_items,
        exchange_delivered_order_items
    ]
