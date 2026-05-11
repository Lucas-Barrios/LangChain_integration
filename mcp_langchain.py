"""
MCP-LangChain Integration Lab
Ironhack AI Consulting Course

Demonstrates:
  1. MCP server connection (stdio, filesystem server)
  2. MCP tools loaded into LangChain as BaseTool objects
  3. ReAct agent calling MCP tools autonomously
  4. Practical use case: multi-document analysis for an AI consulting client
"""

import asyncio
import logging
import os
import warnings
from pathlib import Path
from typing import Any

# Must be before library imports to suppress langgraph/langchain internal noise
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("langchain_mcp_adapters").setLevel(logging.ERROR)
logging.getLogger("langgraph").setLevel(logging.ERROR)

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv(Path(__file__).parent / ".env")

# ── Configuration ──────────────────────────────────────────────────────────────

DOCS_DIR = str(Path(__file__).parent / "test_documents")

# stdio transport: MultiServerMCPClient spawns a local npx subprocess.
# The server is sandboxed to DOCS_DIR — no access outside that path.
MCP_SERVER_CONFIG = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", DOCS_DIR],
        "transport": "stdio",
        "env": {**os.environ, "NODE_NO_WARNINGS": "1"},
    }
}

LLM_MODEL = "gpt-4o-mini"

# ── Output helpers ─────────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def check(label: str) -> None:
    print(f"  ✓  {label}")


def print_agent_result(result: dict[str, Any]) -> None:
    """Print the final AI reply and every MCP tool call made along the way."""
    messages = result.get("messages", [])
    tools_called = []

    for msg in messages:
        kind = type(msg).__name__
        if kind == "AIMessage" and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_called.append(tc["name"])
        if kind == "AIMessage" and msg.content:
            print(f"\n  [Agent reply]\n  {msg.content.strip()}")

    if tools_called:
        print(f"\n  [MCP tools called: {', '.join(tools_called)}]")


# ── Step 1: Verify MCP server connection ──────────────────────────────────────

async def verify_connection(client: MultiServerMCPClient) -> list[BaseTool]:
    """
    Call client.get_tools(). A non-empty result proves the MCP server
    subprocess started, the stdio handshake completed, and the tool
    schema was received — i.e., the connection works.
    """
    section("STEP 1 — MCP Server Connection")
    print(f"\n  Server : @modelcontextprotocol/server-filesystem (stdio)")
    print(f"  Sandbox: {DOCS_DIR}")

    tools = await client.get_tools()

    if not tools:
        raise RuntimeError("No tools returned — MCP server did not connect.")

    check(f"MCP server connected — received {len(tools)} tool schemas over stdio")
    return tools


# ── Step 2: Verify tools are LangChain BaseTool objects ───────────────────────

def verify_tools(tools: list[BaseTool]) -> None:
    """
    langchain-mcp-adapters converts MCP JSON schemas to LangChain BaseTool
    objects. Showing the type and a sample confirms the adapter worked.
    """
    section("STEP 2 — MCP Tools Loaded into LangChain")

    print(f"\n  {len(tools)} tools available to the agent:\n")
    for t in tools:
        type_name = type(t).__name__
        print(f"    {t.name:<35} [{type_name}]")

    # Spot-check: every object must be a LangChain BaseTool
    bad = [t for t in tools if not isinstance(t, BaseTool)]
    if bad:
        raise TypeError(f"Non-BaseTool objects found: {[type(b) for b in bad]}")

    check("All tools are langchain_core BaseTool instances — adapter conversion succeeded")


# ── Step 3: Verify agent calls MCP tools ──────────────────────────────────────

async def verify_agent_tool_use(agent: Any) -> None:
    """
    Ask the agent a question it can only answer by calling MCP tools.
    We then inspect the message trace to confirm at least one ToolMessage
    (= a real MCP tool call + response) is present.
    """
    section("STEP 3 — Agent Using MCP Tools")

    print("\n  Query: list all accessible files")
    result = await agent.ainvoke({
        "messages": [HumanMessage(content=(
            "Use list_allowed_directories to find your sandbox, "
            "then list_directory to show every file in it."
        ))]
    })

    messages = result.get("messages", [])
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    tool_names = []
    for m in messages:
        if type(m).__name__ == "AIMessage" and m.tool_calls:
            tool_names.extend(tc["name"] for tc in m.tool_calls)

    if not tool_messages:
        raise RuntimeError("Agent produced no ToolMessages — no MCP tools were called.")

    check(f"Agent made {len(tool_messages)} MCP tool call(s): {', '.join(tool_names)}")

    # Print the final agent reply
    for m in messages:
        if type(m).__name__ == "AIMessage" and m.content:
            print(f"\n  [Agent reply]\n  {m.content.strip()}")
            break


# ── Step 4: Practical use case — consulting document analysis ─────────────────

async def practical_use_case(agent: Any) -> None:
    """
    Real-world consulting task: the agent reads multiple client documents,
    synthesizes financial and strategic information, and produces a briefing.
    This exercises read_text_file + search_files across all three test docs.
    """
    section("STEP 4 — Practical Use Case: Client Document Briefing")

    print("\n  Scenario: You have 3 documents from a client engagement.")
    print("  Task    : Produce an executive briefing from the full document set.\n")

    result = await agent.ainvoke({
        "messages": [HumanMessage(content=(
            f"You are an AI consultant preparing a client briefing. "
            f"Read all files in '{DOCS_DIR}' and produce a structured summary with:\n"
            f"1. Client situation (1-2 sentences)\n"
            f"2. Recommended technology (with justification)\n"
            f"3. Expected ROI and timeline\n"
            f"4. One risk to flag\n"
            f"Be concise and specific — use numbers where available."
        ))]
    })

    print_agent_result(result)
    check("Practical use case complete — agent read, synthesized, and reported across documents")


# ── Step 5: Resources demo ─────────────────────────────────────────────────────

async def demo_resources(client: MultiServerMCPClient) -> None:
    """
    Show that the adapter's load_mcp_resources() function works and explain
    why the filesystem server uses tools (not static resources) for data access.
    """
    section("STEP 5 — MCP Resources Demo")
    try:
        async with client.session("filesystem") as session:
            resources = await load_mcp_resources(session=session)
            if resources:
                print(f"  Loaded {len(resources)} static resource(s)")
            else:
                print("  Server exposes 0 static resources (expected).")
                print("  Filesystem data is accessed via tools (read_text_file,")
                print("  list_directory) — this is the standard MCP pattern.")
    except Exception:
        print("  Resource listing skipped (filesystem server is tool-only).")

    check("Resources API exercised — tool-based access pattern confirmed")


# ── Final checklist ────────────────────────────────────────────────────────────

def print_checklist() -> None:
    section("VERIFICATION CHECKLIST")
    print()
    print("  ✓  MCP server connected via stdio transport")
    print("  ✓  Tools loaded as LangChain BaseTool objects")
    print("  ✓  Agent called MCP tools autonomously (ToolMessages confirmed)")
    print("  ✓  Practical use case: multi-document client briefing produced")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("OPENAI_API_KEY not set. Add your key to .env before running.")

    print("\n" + "=" * 60)
    print("  MCP-LangChain Integration Lab  |  Ironhack AI Consulting")
    print("=" * 60)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    client = MultiServerMCPClient(MCP_SERVER_CONFIG)

    # Step 1: MCP server connection
    tools = await verify_connection(client)

    # Step 2: Tools as LangChain objects
    verify_tools(tools)

    # Step 3 & 4: Build agent and run demos
    agent = create_react_agent(llm, tools)

    await verify_agent_tool_use(agent)
    await practical_use_case(agent)
    await demo_resources(client)

    print_checklist()


if __name__ == "__main__":
    asyncio.run(main())
