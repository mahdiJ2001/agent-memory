import argparse
import json
import sys

from agent_memory import db, memory, tools
from agent_memory.llm import create_chat, get_client

sys.stdout.reconfigure(encoding="utf-8")


def _augment_with_memories(customer_id: int, user_input: str) -> str:
    facts = memory.retrieve_semantic_memories(customer_id, user_input)
    episodes = memory.retrieve_episodic_memories(customer_id, user_input)

    blocks = []
    if facts:
        print(f"  [memory] Retrieved {len(facts)} fact(s): {facts}")
        facts_block = "\n".join(f"- {fact}" for fact in facts)
        blocks.append(f"[Known facts about this customer]\n{facts_block}")
    if episodes:
        print(f"  [memory] Retrieved {len(episodes)} episode(s): {episodes}")
        episodes_block = "\n".join(f"- {e['event']} -> {e['outcome']}" for e in episodes)
        blocks.append(f"[Past experiences with this customer]\n{episodes_block}")

    procedure = db.get_procedure("recommend_running_shoes")
    if procedure:
        print(f"  [memory] Using procedure '{procedure['name']}' (updated {procedure['updated_at']})")
        steps_block = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(procedure["steps"]))
        blocks.append(f"[Procedure to follow when recommending running shoes]\n{steps_block}")

    if not blocks:
        return user_input

    return "\n\n".join(blocks) + f"\n\n[Customer message]\n{user_input}"


def _resolve_function_calls(chat, response):
    while response.function_calls:
        tool_results = []
        for call in response.function_calls:
            print(f"  [tool call] {call.name}({call.args})")
            result = tools.call_tool(call.name, call.args)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(result),
                }
            )
        response = chat.send_message(tool_results)
    return response


def run() -> None:
    parser = argparse.ArgumentParser(prog="agent-memory")
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable semantic memory retrieval (baseline agent for comparison).",
    )
    args = parser.parse_args()
    use_memory = not args.no_memory

    db.init_db()

    customer_name = input("Who am I speaking with? ").strip() or "Guest"
    customer = db.get_or_create_customer(customer_name)

    extra_context = (
        f"The current customer is {customer['name']} (customer_id={customer['id']}). "
        "Use this customer_id when placing orders; never ask the user for their id."
    )
    if use_memory:
        extra_context += (
            " When the customer states a durable preference or fact about themselves "
            "(not a one-off request), call remember_user_fact with this customer_id to save it. "
            "When the customer wants to return an order, call return_order with the order_id "
            "and their stated reason. Follow the given procedure when recommending running "
            "shoes. If the customer's feedback implies the procedure itself should permanently "
            "change (not just this one answer), call update_procedure with the full revised steps."
        )

    client = get_client()
    chat = create_chat(client, extra_context=extra_context, include_memory=use_memory)

    mode = "WITH memory" if use_memory else "WITHOUT memory (baseline)"
    print(f"\nShopping Assistant [{mode}] (type 'exit' to quit) - hi {customer['name']}!\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        if use_memory:
            user_input = _augment_with_memories(customer["id"], user_input)
        response = chat.send_message(user_input)
        response = _resolve_function_calls(chat, response)
        print(f"Assistant: {response.text}\n")
