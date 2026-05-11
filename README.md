# MCP-LangChain Integration Lab

**Ironhack AI Consulting Course**

A working demonstration of a LangChain ReAct agent connected to an MCP filesystem server.
The agent discovers tools at runtime, reads documents, and performs multi-document analysis
— all through the standardized MCP interface.

---

## File & Folder Map

```
LangChain_Integration/
│
├── mcp_langchain.py          Main script. Runs 5 verification steps:
│                             connection test → tool loading → agent tool use
│                             → practical use case → resources demo.
│
├── requirements.txt          All Python dependencies (one pip install command).
│
├── .env                      Your OpenAI API key goes here. Never committed.
│
├── .gitignore                Excludes .env, __pycache__, .DS_Store, etc.
│
├── lab_summary.md            One-paragraph trade-off analysis: MCP vs direct API,
│                             when to use each, and key architectural differences.
│
├── README.md                 This file — setup, run instructions, architecture.
│
└── test_documents/           Documents sandboxed to the MCP filesystem server.
    │                         The agent can only read/list files inside here.
    │
    ├── ai_trends_2025.txt        2025 AI trends report (agentic AI, RAG, MCP, etc.)
    ├── client_proposal.txt       RetailCo AI consulting proposal with ROI figures.
    └── mcp_technical_overview.txt  MCP architecture, concepts, and comparison table.
```

---

## How to Run

### Prerequisites

| Requirement | Version | Check |
|---|---|---|
| Python | 3.11+ | `python --version` |
| Node.js | 18+ | `node --version` |
| OpenAI API key | — | [platform.openai.com](https://platform.openai.com) |

> **Anaconda users:** use `/opt/anaconda3/bin/python` instead of `python` in the commands below.

---

### Step 1 — Clone the repo

```bash
git clone https://github.com/Lucas-Barrios/LangChain_integration.git
cd LangChain_integration
```

### Step 2 — Install dependencies

```bash
pip install langchain langchain-openai langchain-mcp-adapters mcp python-dotenv
```

Or pin the exact versions:

```bash
pip install -r requirements.txt
```

### Step 3 — Add your API key

Open `.env` and replace the placeholder:

```
OPENAI_API_KEY=sk-...your-key-here...
```

### Step 4 — Run

```bash
python mcp_langchain.py
```

The first run downloads `@modelcontextprotocol/server-filesystem` via `npx` (cached after that).

---

### Expected output

A successful run prints a checkmark for each requirement:

```
✓  MCP server connected — received 14 tool schemas over stdio
✓  All tools are langchain_core BaseTool instances — adapter conversion succeeded
✓  Agent made 2 MCP tool call(s): list_allowed_directories, list_directory
✓  Practical use case complete — agent read, synthesized, and reported across documents
✓  Resources API exercised — tool-based access pattern confirmed

VERIFICATION CHECKLIST
  ✓  MCP server connected via stdio transport
  ✓  Tools loaded as LangChain BaseTool objects
  ✓  Agent called MCP tools autonomously (ToolMessages confirmed)
  ✓  Practical use case: multi-document client briefing produced
```

---

## How It Works

```
mcp_langchain.py
       │
       │  client = MultiServerMCPClient(MCP_SERVER_CONFIG)
       │       │
       │       └── spawns subprocess:
       │            npx @modelcontextprotocol/server-filesystem ./test_documents
       │                      │
       │            stdio transport (JSON-RPC over stdin/stdout)
       │                      │
       │            Exposes 14 tools: read_text_file, list_directory,
       │                              search_files, write_file, ...
       │
       │  tools = await client.get_tools()
       │       └── converts MCP JSON schemas → LangChain StructuredTool objects
       │
       │  agent = create_react_agent(llm, tools)
       │       └── LangGraph ReAct agent (reason → act → observe loop)
       │
       └── agent.ainvoke({"messages": [HumanMessage(...)]})
              └── LLM selects which MCP tools to call
                  Agent executes them, feeds results back to LLM
                  Final answer returned to caller
```

---

## Key Design Decisions

**Why `MultiServerMCPClient`?**
Single object manages connections to multiple MCP servers simultaneously. Pass a config
dict; the library handles subprocess lifecycle and schema-to-BaseTool conversion.
Use `client.session("server_name")` as an async context manager when you need direct
session access (e.g., loading resources).

**Why `stdio` transport?**
The filesystem server runs locally — no HTTP server or network setup needed.
For remote servers (cloud APIs, databases), switch to `"transport": "sse"` with a `"url"`.

**Why `gpt-4o-mini`?**
Cost-effective for lab use. Change `LLM_MODEL` at the top of `mcp_langchain.py`
to `"gpt-4o"` for stronger multi-document reasoning in production.

**Why ReAct agent?**
`create_react_agent` from LangGraph produces a transparent think → act → observe loop.
You can inspect every tool call in the message trace, which makes it ideal for learning.

---

## Customising

**Add documents:** Drop any `.txt` file into `test_documents/` — the agent discovers
it automatically via `list_directory`.

**Add a second MCP server:**
```python
MCP_SERVER_CONFIG = {
    "filesystem": { ... },           # existing
    "database": {                    # new — agent gets tools from both
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sqlite", "mydb.sqlite"],
        "transport": "stdio",
    }
}
```

**Use a remote MCP server (SSE transport):**
```python
"remote_api": {
    "url": "https://your-mcp-server.com/sse",
    "transport": "sse",
}
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: langchain_openai` | Run with `/opt/anaconda3/bin/python` — system Python 3.9 is too old |
| `ValueError: OPENAI_API_KEY not set` | Edit `.env` — placeholder value still present |
| `npx` slow on first run | Normal — package downloads once then is cached by npm |
| `Access denied - path outside allowed directories` | Pass the full `DOCS_DIR` path in your query, not a relative filename |
| Agent takes many steps | Expected — ReAct agents reason step-by-step for multi-file queries |
