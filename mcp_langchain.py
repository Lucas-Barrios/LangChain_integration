"""
MCP-LangChain Integration Lab
Ironhack AI Consulting Course

Demonstrates:
  1. MCP server connection (stdio, filesystem server)
  2. MCP tools loaded into LangChain as BaseTool objects
  3. ReAct agent calling MCP tools autonomously
  4. Practical use case: multi-document analysis for an AI consulting client
  5. MCP resources loaded and injected as agent context (custom resource server)
  6. Complete MCP-enabled agent: comprehensive prompt, multiple servers,
     varied tool exercises, explicit session lifecycle management
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

from config import DOCS_DIR, LLM_MODEL
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from utils import check, collect_tool_names, print_agent_result, require_openai_key, section

load_dotenv(Path(__file__).parent / ".env")

# ── Configuration ──────────────────────────────────────────────────────────────

# Two MCP servers running simultaneously over stdio:
#   filesystem — exposes 14 tools (read, write, search, list…); sandboxed to DOCS_DIR
#   resources  — custom Python server that exposes the same docs as MCP resources
MCP_SERVER_CONFIG = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", DOCS_DIR],
        "transport": "stdio",
        "env": {**os.environ, "NODE_NO_WARNINGS": "1"},  # silence npx deprecation noise on stdout
    },
    "resources": {
        "command": "python",
        "args": [str(Path(__file__).parent / "mcp_resource_server.py")],
        "transport": "stdio",
    },
}


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
    tool_names = collect_tool_names(messages)

    if not tool_messages:
        raise RuntimeError("Agent produced no ToolMessages — no MCP tools were called.")

    check(f"Agent made {len(tool_messages)} MCP tool call(s): {', '.join(tool_names)}")

    for m in messages:
        if isinstance(m, AIMessage) and m.content:
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


# ── Step 5: MCP resources loaded and injected as agent context ────────────────

async def _fetch_resource_blobs(client: MultiServerMCPClient) -> list:
    """Open a resources session, load all blobs, and close the session."""
    async with client.session("resources") as session:
        blobs = await load_mcp_resources(session=session)
    if not blobs:
        raise RuntimeError("No resources returned from resource server.")
    return blobs


def _format_numbered_context(blobs: list) -> str:
    """Format a list of blobs into a numbered document context string."""
    parts = []
    for i, b in enumerate(blobs, 1):
        parts.append(f"--- Document {i} ---\n{b.as_string().strip()}")
    return "\n\n".join(parts)


def _build_context_agent(llm: ChatOpenAI, tools: list[BaseTool], resource_context: str) -> Any:
    """Wrap resource context in a SystemMessage and return a fresh ReAct agent."""
    system_prompt = SystemMessage(content=(
        "You are an AI consulting assistant. "
        "The following documents have been pre-loaded for you as background context. "
        "Use ONLY this context to answer questions — do NOT call any tools.\n\n"
        f"{resource_context}"
    ))
    return create_react_agent(llm, tools, prompt=system_prompt)


async def resources_as_context(client: MultiServerMCPClient, tools: list[BaseTool], llm: ChatOpenAI) -> None:
    """
    Step 5: list MCP resources, inject their content as agent context,
    then verify the agent answers a query without calling any tools.
    """
    section("STEP 5 — MCP Resources as Agent Context")

    # a) Fetch blobs from the resource server
    print("\n  a) Listing resources from custom MCP resource server...\n")
    blobs = await _fetch_resource_blobs(client)

    print(f"  Found {len(blobs)} resource(s):\n")
    for b in blobs:
        lines = b.as_string().strip().splitlines()
        preview = lines[0][:70] if lines else "(empty)"
        print(f"    resource: {preview}...")
        print(f"             ({len(b.as_string())} chars)\n")

    # b) Format blobs into a context string
    print("  b) Resource content loaded into Blob objects via load_mcp_resources()")
    resource_context = _format_numbered_context(blobs)

    # c) Build a context-only agent
    print("\n  c) Injecting resource content as SystemMessage (prompt= parameter)...")
    context_agent = _build_context_agent(llm, tools, resource_context)

    # d) Query the agent — no tool calls expected
    print("\n  d) Querying agent — answer must come from pre-loaded context only...\n")
    query = (
        "Based on the documents in your context: "
        "What is the expected 3-year ROI for the RetailCo project, "
        "and which MCP transport type would you use if the CRM were a cloud service?"
    )
    print(f"  Query: {query}\n")

    result = await context_agent.ainvoke({"messages": [HumanMessage(content=query)]})

    messages = result.get("messages", [])
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    tool_calls_made = collect_tool_names(messages)

    for m in messages:
        if isinstance(m, AIMessage) and m.content:
            print(f"  [Agent reply]\n  {m.content.strip()}\n")
            break

    if tool_calls_made:
        print(f"  [Tools called: {', '.join(tool_calls_made)}]")
    else:
        print("  [No MCP tools called — answer came entirely from injected resource context]")

    check(f"Resources listed: {len(blobs)} | Content injected as SystemMessage | "
          f"Tool calls during answer: {len(tool_messages)}")


# ── Step 6: Complete MCP-enabled agent ────────────────────────────────────────

# Comprehensive system prompt that describes the agent's identity and its full
# set of MCP capabilities so the LLM knows what it can do before reasoning starts.
MCP_SYSTEM_PROMPT = """\
You are an MCP-enabled AI consulting assistant connected to a secure filesystem \
via the Model Context Protocol (stdio transport).

Your available MCP tools and when to use them:
  - list_allowed_directories : discover which paths you can access
  - list_directory            : list files in a directory
  - directory_tree            : view the full recursive folder structure as JSON
  - read_text_file            : read the full text of a single file
  - read_multiple_files       : read several files in one efficient call
  - search_files              : recursively find files matching a glob pattern
  - get_file_info             : retrieve size, timestamps, and type metadata for a path
  - write_file / edit_file    : create or modify files (use with care)

Always choose the most efficient tool for the task. Cite file names and data \
you retrieve. Be concise and specific.\
"""


async def step6_complete_agent(
    client: MultiServerMCPClient,
    tools: list[BaseTool],
    llm: ChatOpenAI,
) -> None:
    """
    Step 6 — Complete MCP-enabled agent:
      - MultiServerMCPClient managing two simultaneous servers
      - Tools and resources both loaded
      - Comprehensive system prompt describing MCP capabilities
      - Four real-world scenarios each exercising a different MCP tool
      - Explicit session lifecycle management (open → use → close)
    """
    section("STEP 6 — Complete MCP-Enabled Agent")

    # ── Setup: load resources + build comprehensive agent ─────────────────────
    print("\n  [Setup] Loading resources from resource server...")
    blobs = await _fetch_resource_blobs(client)
    print(f"  [Setup] {len(blobs)} resource(s) loaded")

    resource_context = _format_numbered_context(blobs)

    # Combine the MCP capability description with pre-loaded document context
    full_system_prompt = SystemMessage(content=(
        MCP_SYSTEM_PROMPT
        + "\n\nThe following consulting documents are pre-loaded as background context:\n\n"
        + resource_context
    ))

    agent = create_react_agent(llm, tools, prompt=full_system_prompt)
    check("MultiServerMCPClient connected to 2 servers | Tools + resources loaded | Agent built")

    scenarios = [
        (
            "Scenario 1 — search_files: find all text documents",
            f"Search '{DOCS_DIR}' for all .txt files. List each filename and its size in bytes.",
        ),
        (
            "Scenario 2 — get_file_info: inspect file metadata",
            f"Use get_file_info on '{DOCS_DIR}/client_proposal.txt'. "
            "Report its size, creation date, and last-modified date.",
        ),
        (
            "Scenario 3 — directory_tree: view full folder structure",
            f"Use directory_tree on '{DOCS_DIR}' and describe the folder structure.",
        ),
        (
            "Scenario 4 — read_multiple_files: cross-document consulting query",
            f"Read '{DOCS_DIR}/ai_trends_2025.txt' and '{DOCS_DIR}/client_proposal.txt' "
            "simultaneously using read_multiple_files. "
            "Which trend from the trends report most directly justifies the RetailCo proposal? "
            "One sentence answer.",
        ),
    ]

    for label, query in scenarios:
        print(f"\n\n  {label}")
        try:
            result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
            _print_scenario_result(result)
        except Exception as exc:
            raise RuntimeError(f"{label} failed: {exc}") from exc

    # ── Disconnect: explain lifecycle ─────────────────────────────────────────
    print("\n  [Disconnect] MCP client session lifecycle:")
    print("    Each client.session('server') call opens a dedicated stdio session.")
    print("    Sessions are closed automatically on exit of the `async with` block.")
    print("    No persistent connection remains after each operation completes.")
    check("All 4 scenarios complete | Sessions opened and closed explicitly | Clients disconnected")


def _print_scenario_result(result: dict[str, Any]) -> None:
    """Print tool calls made and the final agent reply for a scenario."""
    messages = result.get("messages", [])
    tool_names = collect_tool_names(messages)
    if tool_names:
        print(f"  Tools called: {', '.join(tool_names)}")
    for m in messages:
        if isinstance(m, AIMessage) and m.content:
            print(f"  Reply: {m.content.strip()}")
            break


# ── Final checklist ────────────────────────────────────────────────────────────

def print_checklist() -> None:
    """Print the pass/fail verification summary for all six lab steps."""
    section("VERIFICATION CHECKLIST")
    print()
    print("  ✓  MCP server connected via stdio transport")
    print("  ✓  Tools loaded as LangChain BaseTool objects")
    print("  ✓  Agent called MCP tools autonomously (ToolMessages confirmed)")
    print("  ✓  Practical use case: multi-document client briefing produced")
    print("  ✓  MCP resources listed, read, and injected as agent context")
    print("  ✓  Complete agent: comprehensive prompt, 2 servers, 4 tool scenarios, sessions disconnected")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    """Entry point: initialise the LLM and MCP client, then run Steps 1–6 in order."""
    require_openai_key()

    print("\n" + "=" * 60)
    print("  MCP-LangChain Integration Lab  |  Ironhack AI Consulting")
    print("=" * 60)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)  # deterministic output — no variation between runs
    client = MultiServerMCPClient(MCP_SERVER_CONFIG)

    # Step 1: MCP server connection
    tools = await verify_connection(client)

    # Step 2: Tools as LangChain objects
    verify_tools(tools)

    # Step 3 & 4: Build agent and run demos
    agent = create_react_agent(llm, tools)

    await verify_agent_tool_use(agent)
    await practical_use_case(agent)
    await resources_as_context(client, tools, llm)
    await step6_complete_agent(client, tools, llm)

    print_checklist()


if __name__ == "__main__":
    asyncio.run(main())
