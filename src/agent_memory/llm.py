import os

import anthropic
from dotenv import load_dotenv

from agent_memory import tools

load_dotenv()

MODEL_NAME = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a helpful shopping assistant for a small shoe store.
Help customers find products, answer questions, and make recommendations.
Use the available tools to search products, look up details, and place orders
instead of guessing. Be concise and friendly."""


class ToolCall:
    def __init__(self, id: str, name: str, args: dict):
        self.id = id
        self.name = name
        self.args = args


class Response:
    def __init__(self, message: anthropic.types.Message):
        self.message = message
        self.function_calls = [
            ToolCall(block.id, block.name, block.input)
            for block in message.content
            if block.type == "tool_use"
        ]
        text_blocks = [block.text for block in message.content if block.type == "text"]
        self.text = "\n".join(text_blocks)


class Chat:
    def __init__(self, client: anthropic.Anthropic, system: str, tool_declarations: list[dict]):
        self.client = client
        self.system = system
        self.tool_declarations = tool_declarations
        self.messages: list[dict] = []

    def send_message(self, content) -> Response:
        self.messages.append({"role": "user", "content": content})
        message = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=self.system,
            tools=self.tool_declarations,
            messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": message.content})
        return Response(message)


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Check your .env file.")
    return anthropic.Anthropic(api_key=api_key)


def create_chat(client: anthropic.Anthropic, extra_context: str = "", include_memory: bool = True) -> Chat:
    system = SYSTEM_PROMPT
    if extra_context:
        system = f"{SYSTEM_PROMPT}\n\n{extra_context}"
    return Chat(client, system, tools.get_tools(include_memory))
