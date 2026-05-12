"""
Web demo interface for the MCP Document Analysis Agent.

Run with:
    uvicorn web_app:app --reload --port 8000
Then open: http://localhost:8000
"""

import asyncio
import json
import os
import warnings
from pathlib import Path
from typing import AsyncGenerator

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import DOCS_DIR, LLM_MODEL

app = FastAPI(title="MCP Document Analysis Agent Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_SYSTEM_PROMPT = SystemMessage(
    content="""\
You are a Document Analysis Agent — an AI specialist in professional \
document intelligence for consulting engagements.

Available MCP filesystem tools:
  list_allowed_directories, list_directory, search_files,
  get_file_info, read_text_file, read_multiple_files, directory_tree

Output standards:
  • Cite exact filenames. Quote figures verbatim.
  • Use read_multiple_files for 2+ documents.
  • Structure answers with clear headings. Be concise.\
"""
)

_USE_CASES: dict[int, dict] = {
    1: {
        "title": "Document Discovery",
        "query": (
            f"Search '{DOCS_DIR}' for all .txt files using search_files. "
            f"Then call get_file_info on each file found. "
            f"Present results as a structured inventory: "
            f"filename | size | last-modified date | one-sentence description."
        ),
    },
    2: {
        "title": "Targeted Data Extraction",
        "query": (
            f"Read '{DOCS_DIR}/client_proposal.txt' using read_text_file. "
            f"Extract and present as clearly labelled fields:\n"
            f"• Client name and profile\n"
            f"• Core problem being solved (with key metrics)\n"
            f"• Every financial figure mentioned (quote verbatim)\n"
            f"• Project phases with timelines\n"
            f"• Recommended technology stack"
        ),
    },
    3: {
        "title": "Executive Summary",
        "query": (
            f"Use read_multiple_files in a single call to read all three documents: "
            f"'{DOCS_DIR}/ai_trends_2025.txt', '{DOCS_DIR}/client_proposal.txt', "
            f"'{DOCS_DIR}/mcp_technical_overview.txt'. "
            f"Produce a C-suite executive briefing with exactly these sections:\n"
            f"1. Situation — what problem the client faces and why it matters now (2 sentences)\n"
            f"2. Approach — recommended solution and enabling technology (2 sentences)\n"
            f"3. Value Case — expected ROI, timeline, and key metrics (bullet points)\n"
            f"4. Top Risk — the single biggest risk and one mitigation step (1-2 sentences)\n"
            f"Cite the source filename for each section."
        ),
    },
    4: {
        "title": "Cross-Document Search",
        "query": (
            f"A client stakeholder is questioning whether MCP is the right integration "
            f"architecture. Build an evidence brief. Use read_multiple_files to read all "
            f"three documents at once: '{DOCS_DIR}/ai_trends_2025.txt', "
            f"'{DOCS_DIR}/client_proposal.txt', '{DOCS_DIR}/mcp_technical_overview.txt'. "
            f"Search every document for mentions of MCP, Model Context Protocol, or "
            f"protocol/integration architecture decisions. Compile a sourced evidence brief:\n"
            f"• What each document says about MCP (one paragraph per document, cite filename)\n"
            f"• A one-sentence conclusion: does the evidence support using MCP for RetailCo?"
        ),
    },
}


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def stream_analysis(use_case_id: int) -> AsyncGenerator[str, None]:
    uc = _USE_CASES.get(use_case_id)
    if not uc:
        yield sse({"type": "error", "message": f"Unknown use case: {use_case_id}"})
        return

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_openai_api_key_here":
        yield sse({"type": "error", "message": "OPENAI_API_KEY not configured. Check your .env file."})
        return

    yield sse({"type": "status", "message": "Connecting to MCP filesystem server..."})
    await asyncio.sleep(0)

    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        client = MultiServerMCPClient(
            {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", DOCS_DIR],
                    "transport": "stdio",
                    "env": {**os.environ, "NODE_NO_WARNINGS": "1"},
                }
            }
        )

        tools = await client.get_tools()
        yield sse({"type": "status", "message": f"MCP server connected — {len(tools)} tools loaded"})
        await asyncio.sleep(0)

        agent = create_react_agent(llm, tools, prompt=_SYSTEM_PROMPT)
        yield sse({"type": "status", "message": "Agent initialized. Running analysis..."})
        await asyncio.sleep(0)

        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=uc["query"])]},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_tool_start":
                yield sse({"type": "tool_start", "tool": event.get("name", "tool")})

            elif kind == "on_tool_end":
                yield sse({"type": "tool_end", "tool": event.get("name", "tool")})

            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    if not chunk.tool_call_chunks:
                        yield sse({"type": "token", "content": chunk.content})

        yield sse({"type": "done"})

    except Exception as exc:
        yield sse({"type": "error", "message": str(exc)})


@app.get("/api/run/{use_case_id}")
async def run_analysis(use_case_id: int):
    return StreamingResponse(
        stream_analysis(use_case_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/use-cases")
async def get_use_cases():
    return [{"id": k, "title": v["title"]} for k, v in _USE_CASES.items()]


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")
