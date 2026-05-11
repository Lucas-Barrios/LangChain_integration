# MCP-LangChain Integration Lab

**Ironhack AI Consulting Course**

A working demonstration of connecting a LangChain ReAct agent to an MCP (Model Context Protocol)
filesystem server. The agent dynamically discovers tools, reads documents, and performs
multi-document analysis — all through the standardized MCP interface.

---

## What This Lab Demonstrates

| Capability | How it's shown |
|---|---|
| MCP tool discovery | Agent loads tools dynamically at runtime from the filesystem server |
| MCP resource access | Direct resource loading via `load_mcp_resources` |
| Agent + MCP tools | ReAct agent uses `read_file`, `list_directory`, `search_files` |
| Document analysis | Agent reads and summarizes `.txt` files |
| Cross-document synthesis | Agent reads multiple files and connects information across them |
| Content search | Agent searches for specific terms across all documents |

---

## Project Structure

```
LangChain_Integration/
├── mcp_langchain.py          # Main script — all demos
├── requirements.txt          # Python dependencies
├── .env                      # API key (you must fill this in)
├── README.md                 # This file
├── lab_summary.md            # MCP vs direct API trade-off analysis + sample output
└── test_documents/           # Documents served by the MCP filesystem server
    ├── ai_trends_2025.txt
    ├── client_proposal.txt
    └── mcp_technical_overview.txt
```

---

## Setup

### Prerequisites

- Python 3.11+ (Anaconda base env works)
- Node.js 18+ (for the MCP filesystem server via `npx`)
- An OpenAI API key

### 1. Install Python dependencies

```bash
pip install langchain langchain-openai langchain-mcp-adapters mcp python-dotenv
```

### 2. Set your OpenAI API key

Edit `.env`:
```
OPENAI_API_KEY=sk-...your-key-here...
```

### 3. Run

```bash
# Using Anaconda Python (recommended):
/opt/anaconda3/bin/python mcp_langchain.py

# Or if your default python is 3.11+:
python mcp_langchain.py
```

The first run will auto-download `@modelcontextprotocol/server-filesystem` via `npx` (~2 seconds).

---

## How It Works

```
mcp_langchain.py
       │
       │  async with MultiServerMCPClient(config)
       │       │
       │       └── launches subprocess:
       │            npx @modelcontextprotocol/server-filesystem ./test_documents
       │                      │
       │            MCP stdio transport (JSON-RPC over stdin/stdout)
       │                      │
       │            Exposes tools: read_file, list_directory,
       │                           search_files, write_file, ...
       │
       │  tools = await client.get_tools()
       │       └── converts MCP schemas → LangChain BaseTool objects
       │
       │  agent = create_react_agent(llm, tools)
       │       └── LangGraph ReAct agent (think → act → observe loop)
       │
       └── agent.ainvoke({"messages": [HumanMessage(...)]})
              └── LLM decides which tools to call, agent executes them
                  via MCP server, result fed back to LLM
```

---

## Key Design Decisions

**Why `MultiServerMCPClient`?**
It manages multiple server connections with a single async context manager, handles
subprocess lifecycle, and converts MCP tool schemas to LangChain format automatically.

**Why `stdio` transport?**
The filesystem server runs locally as a subprocess — no HTTP server setup required.
For remote MCP servers (cloud APIs, databases), use `"transport": "sse"` instead.

**Why `gpt-4o-mini`?**
Cost-effective for lab use. For production document analysis, upgrade to `gpt-4o`
for better multi-document reasoning. Change `LLM_MODEL` in the script.

**Why ReAct agent?**
`create_react_agent` (from LangGraph) gives transparent step-by-step reasoning —
ideal for learning because you can see exactly which tools the agent calls and why.

---

## Customizing

**Add more documents:** Drop any `.txt` file into `test_documents/` — the agent
discovers them dynamically via `list_directory`.

**Connect to a different MCP server:** Add another entry to `MCP_SERVER_CONFIG`:
```python
MCP_SERVER_CONFIG = {
    "filesystem": {...},                    # existing
    "database": {                           # new
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sqlite", "mydb.sqlite"],
        "transport": "stdio",
    }
}
```
The agent now has tools from both servers simultaneously.

**Use a remote MCP server (SSE):**
```python
"remote_api": {
    "url": "https://your-mcp-server.com/sse",
    "transport": "sse",
}
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: langchain_openai` | Use `/opt/anaconda3/bin/python` instead of system Python |
| `ValueError: OPENAI_API_KEY not set` | Edit `.env` and add your key |
| `npx` slow on first run | Normal — it downloads the MCP package once, then caches it |
| Agent takes many steps | Expected for multi-document queries; ReAct agents are thorough |
