"""
Account tools adapted from tau-bench.

4 tools:
- find_user_id_by_email
- find_user_id_by_name_zip
- get_user_details
- modify_user_address
"""

import json
from typing import Annotated
from pydantic import Field
from agent_framework import ai_function

from .base import get_active_database, log_tool_call


@ai_function
def find_user_id_by_email(
    email: Annotated[str, Field(description="The email of the user (must be provided by the user, never fabricate).")]
) -> str:
    """Find user id by email. If the user is not found, the function will return an error message."""
    db = get_active_database()
    users = db.users

    for user_id, profile in users.items():
        if profile["email"].lower() == email.lower():
            log_tool_call("find_user_id_by_email", {"email": email}, user_id)
            return user_id

    result = "Error: user not found"
    log_tool_call("find_user_id_by_email", {"email": email}, result)
    return result


@ai_function
def find_user_id_by_name_zip(
    first_name: Annotated[str, Field(description="The first name of the customer, such as 'John'.")],
    last_name: Annotated[str, Field(description="The last name of the customer, such as 'Doe'.")],
    zip: Annotated[str, Field(description="The zip code of the customer, such as '12345'.")]
) -> str:
    """Find user id by first name, last name, and zip code. If the user is not found, the function will return an error message."""
    db = get_active_database()
    users = db.users

    for user_id, profile in users.items():
        if (
            profile["name"]["first_name"].lower() == first_name.lower()
            and profile["name"]["last_name"].lower() == last_name.lower()
            and profile["address"]["zip"] == zip
        ):
            log_tool_call("find_user_id_by_name_zip",
                          {"first_name": first_name, "last_name": last_name, "zip": zip},
                          user_id)
            return user_id

    result = "Error: user not found"
    log_tool_call("find_user_id_by_name_zip",
                  {"first_name": first_name, "last_name": last_name, "zip": zip},
                  result)
    return result


@ai_function
def get_user_details(
    user_id: Annotated[str, Field(description="The user id, such as 'sara_doe_496'.")]
) -> str:
    """Get the details of a user, including their orders."""
    db = get_active_database()
    users = db.users

    if user_id in users:
        result = json.dumps(users[user_id])
        log_tool_call("get_user_details", {"user_id": user_id}, result)
        return result

    result = "Error: user not found"
    log_tool_call("get_user_details", {"user_id": user_id}, result)
    return result


@ai_function
def modify_user_address(
    user_id: Annotated[str, Field(description="The user id, such as 'sara_doe_496'.")],
    address1: Annotated[str, Field(description="The first line of the address, such as '123 Main St'.")],
    address2: Annotated[str, Field(description="The second line of the address, such as 'Suite 100'.")],
    city: Annotated[str, Field(description="The city, such as 'San Francisco'.")],
    state: Annotated[str, Field(description="The state, such as 'CA'.")],
    country: Annotated[str, Field(description="The country, such as 'USA'.")],
    zip: Annotated[str, Field(description="The zip code, such as '12345'.")]
) -> str:
    """Modify the default address of a user."""
    db = get_active_database()
    users = db.users

    if user_id not in users:
        result = "Error: user not found"
        log_tool_call("modify_user_address", {"user_id": user_id}, result)
        return result

    users[user_id]["address"] = {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip
    }

    result = json.dumps(users[user_id])
    log_tool_call("modify_user_address",
                  {"user_id": user_id, "address1": address1, "city": city},
                  result)
    db.record_mutation("modify_user_address", {"user_id": user_id}, result)
    return result


def get_account_tools() -> list:
    """Return all account tools."""
    return [
        find_user_id_by_email,
        find_user_id_by_name_zip,
        get_user_details,
        modify_user_address
    ]
