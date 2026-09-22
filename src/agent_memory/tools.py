import datetime
import decimal
from typing import Any, Callable

from agent_memory import db, memory

_BASE_DECLARATIONS = [
    {
        "name": "search_products",
        "description": "Search the product catalog by keyword (matches name, brand, category, or description).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword to search for, e.g. 'running', 'Nike', 'lightweight'.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_product",
        "description": "Get full details for a single product by its id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "integer", "description": "The product id."},
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "create_order",
        "description": "Place an order for a customer buying a given product.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer", "description": "The customer id."},
                "product_id": {"type": "integer", "description": "The product id to order."},
                "quantity": {
                    "type": "integer",
                    "description": "How many units to order. Defaults to 1.",
                },
            },
            "required": ["customer_id", "product_id"],
        },
    },
    {
        "name": "get_order",
        "description": "Look up an existing order by its id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "integer", "description": "The order id."},
            },
            "required": ["order_id"],
        },
    },
]

_REMEMBER_DECLARATION = {
    "name": "remember_user_fact",
    "description": (
        "Save a durable fact or preference the customer stated about themselves, so it can be "
        "used in future conversations (e.g. shoe preferences, typical running distance). Only "
        "call this for information that will still be true weeks from now - not one-off requests "
        "like 'show me Nike shoes' or order-related questions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_id": {"type": "integer", "description": "The customer id this fact belongs to."},
            "fact": {
                "type": "string",
                "description": "A concise statement of the fact, e.g. 'User prefers lightweight running shoes.'",
            },
        },
        "required": ["customer_id", "fact"],
    },
}


_RETURN_ORDER_DECLARATION = {
    "name": "return_order",
    "description": (
        "Process a return for an existing order. Call this when the customer wants to return "
        "a product they ordered and explains why (e.g. wrong fit, damaged, changed their mind)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {"type": "integer", "description": "The order id being returned."},
            "reason": {
                "type": "string",
                "description": "Why the customer is returning it, e.g. 'felt too narrow'.",
            },
        },
        "required": ["order_id", "reason"],
    },
}


_UPDATE_PROCEDURE_DECLARATION = {
    "name": "update_procedure",
    "description": (
        "Revise the stored step-by-step procedure for a task, when the customer's feedback "
        "implies the process itself should permanently change - not just this one answer. "
        "For example, if they say you don't need to ask a certain question every time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The procedure name, e.g. 'recommend_running_shoes'.",
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The full revised ordered list of steps, replacing the old ones.",
            },
        },
        "required": ["name", "steps"],
    },
}


def get_tools(include_memory: bool = True) -> list[dict]:
    declarations = list(_BASE_DECLARATIONS)
    if include_memory:
        declarations.append(_REMEMBER_DECLARATION)
        declarations.append(_RETURN_ORDER_DECLARATION)
        declarations.append(_UPDATE_PROCEDURE_DECLARATION)
    return declarations


def _search_products(query: str) -> dict:
    return {"products": db.search_products(query)}


def _get_product(product_id: int) -> dict:
    product = db.get_product(product_id)
    return {"product": product} if product else {"error": f"No product with id {product_id}"}


def _create_order(customer_id: int, product_id: int, quantity: int = 1) -> dict:
    order = db.create_order(customer_id, product_id, quantity)
    return {"order": order} if order else {"error": "Could not create order"}


def _get_order(order_id: int) -> dict:
    order = db.get_order(order_id)
    return {"order": order} if order else {"error": f"No order with id {order_id}"}


def _remember_user_fact(customer_id: int, fact: str) -> dict:
    row = memory.add_semantic_memory(customer_id, fact)
    return {"saved": True, "memory_id": row["id"], "fact": row["content"]}


def _return_order(order_id: int, reason: str) -> dict:
    order = db.get_order(order_id)
    if not order:
        return {"error": f"No order with id {order_id}"}

    db.update_order_status(order_id, "returned")

    event = f"Bought {order['product_name']}"
    outcome = f"Returned - {reason}"
    memory.add_episodic_memory(order["customer_id"], order_id, event, outcome)

    return {
        "order_id": order_id,
        "status": "returned",
        "episode_recorded": {"event": event, "outcome": outcome},
    }


def _update_procedure(name: str, steps: list[str]) -> dict:
    row = db.update_procedure(name, steps)
    return {"updated": True, "name": row["name"], "steps": row["steps"]}


DISPATCH: dict[str, Callable[..., dict]] = {
    "search_products": _search_products,
    "get_product": _get_product,
    "create_order": _create_order,
    "get_order": _get_order,
    "remember_user_fact": _remember_user_fact,
    "return_order": _return_order,
    "update_procedure": _update_procedure,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def call_tool(name: str, args: dict) -> dict:
    if name not in DISPATCH:
        return {"error": f"Unknown tool: {name}"}
    result = DISPATCH[name](**args)
    return _json_safe(result)
