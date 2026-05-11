"""
MCP-LangChain Integration Lab
Ironhack AI Consulting Course

Demonstrates connecting a LangChain ReAct agent to an MCP filesystem server,
loading tools dynamically, reading resources, and performing document analysis.
"""

import asyncio
import logging
import os
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Suppress verbose deprecation warnings from langgraph/langchain internals
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("langchain_mcp_adapters").setLevel(logging.ERROR)
logging.getLogger("langgraph").setLevel(logging.ERROR)
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.resources import load_mcp_resources
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Load .env from the same directory as this script regardless of cwd
load_dotenv(Path(__file__).parent / ".env")

# ── Configuration ──────────────────────────────────────────────────────────────

DOCS_DIR = str(Path(__file__).parent / "test_documents")

# MCP server config: stdio transport launches a subprocess.
# @modelcontextprotocol/server-filesystem exposes read_file, write_file,
# list_directory, search_files, etc. as MCP tools.
MCP_SERVER_CONFIG = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", DOCS_DIR],
        "transport": "stdio",
        # Redirect MCP server's own stderr so it doesn't pollute lab output
        "env": {**os.environ, "NODE_NO_WARNINGS": "1"},
    }
}

LLM_MODEL = "gpt-4o-mini"  # cost-effective for a lab; swap to gpt-4o for production


# ── Helper: pretty-print agent response ────────────────────────────────────────

def print_response(label: str, result: dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    messages = result.get("messages", [])
    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        # Only print human and AI messages to keep output clean
        if role in ("Human", "AI", "AIMessage"):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content:
                prefix = "User" if role == "Human" else "Agent"
                print(f"\n[{prefix}]\n{content}")
    print()


# ── Demo 1: List available tools ───────────────────────────────────────────────

async def demo_list_tools(client: MultiServerMCPClient) -> list:
    """Load and display all MCP tools exposed by the filesystem server."""
    print("\n--- Phase 1: Loading MCP Tools ---")
    tools = await client.get_tools()
    print(f"Found {len(tools)} tools from MCP server:\n")
    for tool in tools:
        print(f"  Tool: {tool.name}")
        print(f"    Description: {tool.description[:80]}...")
        print()
    return tools


# ── Demo 2: Read MCP resources directly ────────────────────────────────────────

async def demo_read_resources(client: MultiServerMCPClient) -> None:
    """
    Read MCP resources directly (bypassing the agent).
    Resources are data streams exposed by the server, distinct from tool calls.
    Uses client.session() context manager — the v0.1.0+ API.
    """
    print("\n--- Phase 2: Reading MCP Resources Directly ---")
    try:
        async with client.session("filesystem") as session:
            resources = await load_mcp_resources(session=session)
            if not resources:
                print("  (Server exposes no static resources — tools are the interface)")
            else:
                print(f"  Loaded {len(resources)} resource(s):")
                for r in resources:
                    print(f"    URI: {r.metadata.get('source', 'unknown')}")
    except Exception as exc:
        # Filesystem server uses tools rather than resources; this is expected.
        print(f"  Note: resource listing returned: {type(exc).__name__}")
        print("  The filesystem MCP server exposes data via tools (read_file, list_directory)")
        print("  rather than static resources — this is the standard pattern.\n")


# ── Demo 3: Agent lists directory ──────────────────────────────────────────────

async def demo_list_directory(agent: Any) -> None:
    """Agent uses list_allowed_directories + list_directory to discover documents."""
    print("\n--- Phase 3: Agent Discovers Documents via MCP Tool ---")
    result = await agent.ainvoke({
        "messages": [HumanMessage(content=(
            "First use list_allowed_directories to find out which directories you can access. "
            "Then use list_directory on that path to show me all files available. "
            "List each file name clearly."
        ))]
    })
    print_response("Agent: List Directory", result)


# ── Demo 4: Document analysis ──────────────────────────────────────────────────

async def demo_analyze_document(agent: Any, docs_dir: str) -> None:
    """Agent reads and analyzes a specific document using read_text_file."""
    print("\n--- Phase 4: Agent Reads & Analyzes a Document ---")
    result = await agent.ainvoke({
        "messages": [HumanMessage(content=(
            f"Read the file at '{docs_dir}/ai_trends_2025.txt' and give me a 3-bullet "
            "executive summary of the top AI trends, written for a non-technical business audience."
        ))]
    })
    print_response("Agent: Document Analysis", result)


# ── Demo 5: Cross-document synthesis ──────────────────────────────────────────

async def demo_cross_document_search(agent: Any, docs_dir: str) -> None:
    """Agent reads multiple documents and synthesizes a cross-document answer."""
    print("\n--- Phase 5: Agent Cross-Document Synthesis ---")
    result = await agent.ainvoke({
        "messages": [HumanMessage(content=(
            f"Read '{docs_dir}/client_proposal.txt' and '{docs_dir}/mcp_technical_overview.txt'. "
            "Explain how MCP specifically enables the RetailCo solution described in the proposal. "
            "Be specific about which MCP capabilities map to which proposal components."
        ))]
    })
    print_response("Agent: Cross-Document Synthesis", result)


# ── Demo 6: Search within documents ───────────────────────────────────────────

async def demo_search_content(agent: Any, docs_dir: str) -> None:
    """Agent uses search_files + read_file to find specific content across documents."""
    print("\n--- Phase 6: Agent Searches for Specific Content ---")
    result = await agent.ainvoke({
        "messages": [HumanMessage(content=(
            f"Search in the directory '{docs_dir}' for any files containing financial information "
            "such as ROI, cost, investment, or dollar amounts. "
            "Read the relevant files and list every financial figure you find, noting which file each came from."
        ))]
    })
    print_response("Agent: Content Search", result)


# ── Main orchestrator ──────────────────────────────────────────────────────────

async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY not set. Add your key to .env before running."
        )

    print("\n" + "="*60)
    print("  MCP-LangChain Integration Lab")
    print("  Ironhack AI Consulting Course")
    print("="*60)
    print(f"\nDocuments directory: {DOCS_DIR}")
    print(f"MCP server: @modelcontextprotocol/server-filesystem (stdio)")
    print(f"LLM: {LLM_MODEL} via ChatOpenAI\n")

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    # v0.1.0+ API: MultiServerMCPClient is NOT an async context manager.
    # Create it directly and use client.session(name) for session-scoped work.
    client = MultiServerMCPClient(MCP_SERVER_CONFIG)

    # Phase 1: Inspect available tools (client manages sessions internally)
    tools = await demo_list_tools(client)

    # Phase 2: Try reading MCP resources directly via a named session
    await demo_read_resources(client)

    # Build the ReAct agent — tools are plain LangChain BaseTool objects
    # converted from MCP tool schemas by the adapter library.
    agent = create_react_agent(llm, tools)
    print("\n[Agent ready — ReAct agent with MCP filesystem tools]\n")

    # Phases 3-6: Run document analysis demos
    await demo_list_directory(agent)
    await demo_analyze_document(agent, DOCS_DIR)
    await demo_cross_document_search(agent, DOCS_DIR)
    await demo_search_content(agent, DOCS_DIR)

    print("\n" + "="*60)
    print("  Lab complete. All MCP capabilities demonstrated.")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
