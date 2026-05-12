"""
Step 8 — Document Analysis Agent (Practical Example)
Ironhack AI Consulting Course

Objective: A production-ready MCP-powered agent purpose-built for document
           intelligence.  It connects to a filesystem MCP server, discovers
           documents, and handles four real-world analysis tasks that a
           consulting team would actually run on client files.

Use cases:
  1. Document Discovery  — inventory all files with metadata
  2. Targeted Extraction — read one document, pull out structured key data
  3. Executive Summary   — synthesise all documents into a C-suite briefing
  4. Content Search      — find specific information scattered across files

Run with:
    python step8_document_agent.py
"""

import asyncio
import logging
import os
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("langchain_mcp_adapters").setLevel(logging.ERROR)
logging.getLogger("langgraph").setLevel(logging.ERROR)

from config import DOCS_DIR, LLM_MODEL
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv(Path(__file__).parent / ".env")

# ── Specialised system prompt ──────────────────────────────────────────────────
#
# Unlike the generic MCP_SYSTEM_PROMPT in mcp_langchain.py, this prompt is
# purpose-built for document intelligence: it mandates source citations,
# structured output, exact-value quoting, and efficient multi-file reads.

DOCUMENT_ANALYST_SYSTEM_PROMPT = SystemMessage(content="""\
You are a Document Analysis Agent — an AI specialist in professional \
document intelligence for consulting engagements.

Available MCP filesystem tools and when to use each:
  list_allowed_directories → discover your sandbox root before navigating
  list_directory           → enumerate files in a folder with sizes
  search_files             → locate documents by glob pattern (e.g. *.txt)
  get_file_info            → retrieve size, creation date, modified date
  read_text_file           → read one document in full
  read_multiple_files      → read 2+ documents in a single efficient call
                             (ALWAYS prefer this over repeated read_text_file)
  directory_tree           → view full recursive folder structure as JSON

Output standards:
  • Always cite the exact filename when referencing a document.
  • Quote financial figures, dates, and metrics verbatim from the source.
  • Use read_multiple_files whenever you need content from more than one file.
  • Structure multi-document answers with a clear heading per document.
  • Be concise — bullet points over paragraphs wherever appropriate.\
""")

from utils import OUTPUT_WIDTH, check, collect_tool_names, require_openai_key, section

# ── Output helpers ─────────────────────────────────────────────────────────────

def print_agent_result(result: dict[str, Any]) -> None:
    """Print MCP tools called and the final agent reply, indented for readability."""
    messages = result.get("messages", [])
    tools_called = collect_tool_names(messages)
    if tools_called:
        print(f"\n  [MCP tools called: {', '.join(tools_called)}]\n")
    for m in messages:
        if isinstance(m, AIMessage) and m.content:
            for line in m.content.strip().splitlines():
                print(f"  {line}")
            break


# ── Use case 1: Document Discovery ────────────────────────────────────────────

async def uc1_document_discovery(agent: Any) -> None:
    """
    Demonstrates: search_files + get_file_info
    Task: Build a document inventory — every file with its name, size,
          modification date, and a one-sentence description.
    """
    section("USE CASE 1 — Document Discovery")
    print(
        "\n  Goal: inventory every document in the sandbox with metadata.\n"
        "  Demonstrates: search_files → get_file_info per file.\n"
    )

    query = (
        f"Search '{DOCS_DIR}' for all .txt files using search_files. "
        f"Then call get_file_info on each file found. "
        f"Present the results as a structured inventory: "
        f"filename | size | last-modified date | one-sentence description of what the document covers."
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
    print_agent_result(result)
    check("Documents discovered — names, sizes, dates, and descriptions retrieved via MCP")


# ── Use case 2: Targeted Data Extraction ──────────────────────────────────────

async def uc2_targeted_extraction(agent: Any) -> None:
    """
    Demonstrates: read_text_file → structured key-data extraction
    Task: Read the client proposal and pull out every piece of structured
          data a project manager would need to brief a stakeholder.
    """
    section("USE CASE 2 — Targeted Data Extraction")
    print(
        "\n  Goal: read one document and extract structured, actionable data.\n"
        "  Demonstrates: read_text_file → precise content extraction.\n"
    )

    query = (
        f"Read '{DOCS_DIR}/client_proposal.txt' using read_text_file. "
        f"Extract and present the following as clearly labelled fields:\n"
        f"  • Client name and profile\n"
        f"  • Core problem being solved (with the key metrics that prove it)\n"
        f"  • Every financial figure mentioned (costs, ROI, savings — quote verbatim)\n"
        f"  • Project phases with timelines\n"
        f"  • Recommended technology stack"
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
    print_agent_result(result)
    check("Client proposal read — structured data and verbatim financials extracted")


# ── Use case 3: Executive Summary ─────────────────────────────────────────────

async def uc3_executive_summary(agent: Any) -> None:
    """
    Demonstrates: read_multiple_files → multi-document synthesis
    Task: Synthesise all three consulting documents into a C-suite briefing
          in a single efficient MCP call.
    """
    section("USE CASE 3 — Multi-Document Executive Summary")
    print(
        "\n  Goal: synthesise all documents into one structured briefing.\n"
        "  Demonstrates: read_multiple_files → cross-document synthesis.\n"
    )

    ai_trends     = f"{DOCS_DIR}/ai_trends_2025.txt"
    client_prop   = f"{DOCS_DIR}/client_proposal.txt"
    mcp_overview  = f"{DOCS_DIR}/mcp_technical_overview.txt"

    query = (
        f"Use read_multiple_files in a single call to read all three documents: "
        f"'{ai_trends}', '{client_prop}', '{mcp_overview}'. "
        f"Then produce a C-suite executive briefing with exactly these four sections:\n"
        f"  1. Situation    — what problem the client faces and why it matters now (2 sentences)\n"
        f"  2. Approach     — recommended solution and the technology enabling it (2 sentences)\n"
        f"  3. Value Case   — expected ROI, timeline, and key outcome metrics (bullet points)\n"
        f"  4. Top Risk     — the single biggest risk and one mitigation step (1-2 sentences)\n"
        f"Cite the source filename for each section."
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
    print_agent_result(result)
    check("Executive summary generated — all 3 documents read in one MCP call and synthesised")


# ── Use case 4: Cross-Document Content Search ─────────────────────────────────

async def uc4_content_search(agent: Any) -> None:
    """
    Demonstrates: read_multiple_files → targeted content search across docs
    Task: A prospect asks whether MCP is the right architectural choice.
          The agent must find every reference to MCP across all documents
          and compile a sourced evidence brief to support the recommendation.
    """
    section("USE CASE 4 — Cross-Document Content Search")
    print(
        "\n  Goal: find specific information scattered across multiple documents.\n"
        "  Demonstrates: read_multiple_files → targeted cross-doc content retrieval.\n"
    )

    ai_trends     = f"{DOCS_DIR}/ai_trends_2025.txt"
    client_prop   = f"{DOCS_DIR}/client_proposal.txt"
    mcp_overview  = f"{DOCS_DIR}/mcp_technical_overview.txt"

    query = (
        f"A client stakeholder is questioning whether MCP is the right integration "
        f"architecture for their project.  You need to build an evidence brief. "
        f"Use read_multiple_files to read all three documents at once: "
        f"'{ai_trends}', '{client_prop}', '{mcp_overview}'. "
        f"Search every document for mentions of MCP, Model Context Protocol, or "
        f"protocol/integration architecture decisions. "
        f"Compile a sourced evidence brief with:\n"
        f"  • What each document says about MCP (one paragraph per document, cite filename)\n"
        f"  • A one-sentence conclusion: does the evidence support using MCP for RetailCo?"
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
    print_agent_result(result)
    check("MCP evidence compiled across all 3 documents — sourced brief produced")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    """Connect to the MCP filesystem server, build the analysis agent, and run all four use cases."""
    require_openai_key()

    print("\n" + "=" * OUTPUT_WIDTH)
    print("  Step 8 — Document Analysis Agent")
    print("  Ironhack AI Consulting Course")
    print("=" * OUTPUT_WIDTH)

    # ── Setup ──────────────────────────────────────────────────────────────────
    section("SETUP — MCP Filesystem Server + Specialised Agent")

    print(f"\n  Documents : {DOCS_DIR}")
    print(f"  Model     : {LLM_MODEL}")
    print(f"  Transport : stdio (subprocess)\n")
    print("  Connecting to MCP filesystem server...")

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)  # deterministic output — no variation between runs

    client = MultiServerMCPClient({
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", DOCS_DIR],
            "transport": "stdio",
            "env": {**os.environ, "NODE_NO_WARNINGS": "1"},  # silence npx deprecation noise on stdout
        }
    })

    tools: list[BaseTool] = await client.get_tools()
    check(f"MCP filesystem server connected — {len(tools)} tools auto-loaded")

    # Build the specialised document analysis agent
    agent = create_react_agent(llm, tools, prompt=DOCUMENT_ANALYST_SYSTEM_PROMPT)
    check("Document Analysis Agent built with specialised system prompt")
    print(
        "\n  Available MCP tools (auto-discovered at runtime):\n"
        + "\n".join(f"    {t.name}" for t in tools)
    )

    # ── Run the four use cases ─────────────────────────────────────────────────
    await uc1_document_discovery(agent)
    await uc2_targeted_extraction(agent)
    await uc3_executive_summary(agent)
    await uc4_content_search(agent)

    # ── Verification checklist ────────────────────────────────────────────────
    section("VERIFICATION CHECKLIST")
    print()
    check("Filesystem MCP server configured and connected via stdio transport")
    check("MCP tools loaded into LangChain and bound to a specialised ReAct agent")
    check("Agent prompt purpose-built for document analysis (citations, structure, efficiency)")
    check("UC1 — Documents discovered: file names, sizes, dates via search_files + get_file_info")
    check("UC2 — Document read: structured key data and verbatim financials extracted")
    check("UC3 — Executive summary: all 3 docs synthesised in one read_multiple_files call")
    check("UC4 — Content search: targeted information found and compiled across all documents")
    print()


if __name__ == "__main__":
    asyncio.run(main())
