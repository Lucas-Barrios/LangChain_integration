"""
Shared output helpers and guards for the MCP-LangChain lab scripts.
"""

import os
from typing import Any

from langchain_core.messages import AIMessage

OUTPUT_WIDTH = 64
_PLACEHOLDER_KEY = "your_openai_api_key_here"


def require_openai_key() -> str:
    """Return the OPENAI_API_KEY or raise a clear ValueError if it is missing or placeholder."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key == _PLACEHOLDER_KEY:
        raise ValueError("OPENAI_API_KEY not set. Add your key to .env before running.")
    return key


def section(title: str) -> None:
    bar = "─" * OUTPUT_WIDTH
    print(f"\n{bar}\n  {title}\n{bar}")


def check(label: str) -> None:
    print(f"  ✓  {label}")


def collect_tool_names(messages: list) -> list[str]:
    """Return the name of every MCP tool called across a message trace."""
    return [
        tc["name"]
        for m in messages
        if isinstance(m, AIMessage) and m.tool_calls
        for tc in m.tool_calls
    ]


def print_agent_result(result: dict[str, Any], indent: str = "  ") -> None:
    """Print the final AI reply and every MCP tool call made along the way."""
    messages = result.get("messages", [])
    tools_called = collect_tool_names(messages)

    for m in messages:
        if isinstance(m, AIMessage) and m.content:
            print(f"\n{indent}[Agent reply]\n{indent}{m.content.strip()}")
            break

    if tools_called:
        print(f"\n{indent}[MCP tools called: {', '.join(tools_called)}]")
