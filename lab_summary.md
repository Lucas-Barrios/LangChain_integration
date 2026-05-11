# Lab Summary: MCP vs Direct API Trade-offs

## MCP vs Direct API Integration — When to Use Each

**Model Context Protocol (MCP)** and **direct API integration** both connect AI agents
to external data sources, but they make fundamentally different architectural bets.

MCP standardizes the interface between agents and tools — one client library connects
to any MCP-compliant server, tools are discovered dynamically at runtime, and swapping
a data source (say, replacing a local filesystem server with a cloud storage server)
requires zero changes to the agent code. This portability is its core value: the same
LangChain agent used in this lab could connect to a database, a web scraper, or a
code execution sandbox simply by changing the server config dict. The trade-off is
indirection — each tool call crosses a subprocess or HTTP boundary, adding latency
(typically 10–50ms per call), and debugging requires understanding both the agent
layer and the MCP server layer separately. For simple, well-defined integrations
where you control both sides (e.g., an internal read-only API you built yourself),
the standardization overhead may not be worth it.

**Direct API integration** — wrapping API calls as LangChain `BaseTool` subclasses
manually — gives you full control: you handle auth, retry logic, response shaping,
and error messages exactly as your use case demands, with no subprocess overhead and
easier local debugging. The cost is that every new data source is a custom
implementation, making multi-source agents harder to maintain as the tool count grows.

**Practical recommendation for AI consulting:** use MCP when building agents that will
connect to many data sources, when the tool ecosystem is likely to change, or when
re-using agents across projects. Use direct integration for narrow, stable use cases
where you need maximum control over latency and error handling.

---

## Key Technical Decisions Made in This Lab

| Decision | Choice | Reason |
|---|---|---|
| MCP transport | `stdio` | Local filesystem server; no network setup required |
| Agent type | ReAct (`create_react_agent`) | Best for tool-heavy reasoning with transparent step trace |
| LLM | `gpt-4o-mini` | Balances capability with cost for a lab setting |
| MCP server | `@modelcontextprotocol/server-filesystem` | Official Anthropic reference implementation |
| Resource access | Via tools (`read_file`) | Filesystem server exposes data through tools, not static resources |

---

## Sample Console Output

```
============================================================
  MCP-LangChain Integration Lab
  Ironhack AI Consulting Course
============================================================

Documents directory: /path/to/test_documents
MCP server: @modelcontextprotocol/server-filesystem (stdio)
LLM: gpt-4o-mini via ChatOpenAI

--- Phase 1: Loading MCP Tools ---
Found 10 tools from MCP server:

  Tool: read_file
    Description: Read the complete contents of a file from the file system...

  Tool: read_multiple_files
    Description: Read the contents of multiple files simultaneously...

  Tool: write_file
    Description: Create a new file or completely overwrite an existing file...

  Tool: edit_file
    Description: Make line-based edits to a text file...

  Tool: create_directory
    Description: Create a new directory or ensure a directory exists...

  Tool: list_directory
    Description: Get a listing of all files and directories in a specified path...

  Tool: directory_tree
    Description: Get a recursive tree view of files and directories...

  Tool: move_file
    Description: Move or rename files and directories...

  Tool: search_files
    Description: Recursively search for files and directories matching a pattern...

  Tool: get_file_info
    Description: Retrieve detailed metadata about a file or directory...

--- Phase 2: Reading MCP Resources Directly ---
  Note: The filesystem MCP server exposes data via tools (read_file, list_directory)
  rather than static resources — this is the standard pattern.

[Agent ready — ReAct agent with MCP filesystem tools]

--- Phase 3: Agent Discovers Documents via MCP Tool ---
[Agent]
I'll list the files in the current directory for you.

The directory contains 3 files:
- ai_trends_2025.txt
- client_proposal.txt
- mcp_technical_overview.txt

--- Phase 4: Agent Reads & Analyzes a Document ---
[Agent]
Here's a 3-bullet executive summary of the top AI trends for a business audience:

• **Agentic AI is here** — AI can now autonomously complete multi-step tasks by calling
  external tools and APIs, dramatically reducing manual work in workflows like customer support,
  data analysis, and research.

• **Every data source can now plug into AI** — The Model Context Protocol (MCP) acts like
  "USB-C for AI," letting businesses connect their existing databases, files, and APIs to
  any AI agent without custom engineering for each integration.

• **AI is getting eyes and ears** — Leading AI models now natively understand images, audio,
  and video, opening new automation opportunities in document processing, quality control,
  and customer-facing media.

--- Phase 5: Agent Cross-Document Synthesis ---
[Agent]
MCP directly enables the RetailCo solution in three concrete ways:

1. **Order Management API connection** (Proposal Phase 2) → MCP tool wrapper lets the
   LangChain agent call order lookup endpoints without custom code per endpoint.

2. **CRM integration** (Proposal Phase 2) → A CRM MCP server exposes customer history
   as tools; the agent can retrieve context mid-conversation without hardcoded API calls.

3. **Knowledge base (RAG)** (Proposal Phase 2) → While typically handled via vector search,
   an MCP resource server can expose chunked KB articles as readable resources, keeping the
   retrieval layer swappable.

--- Phase 6: Agent Searches for Specific Content ---
[Agent]
Financial figures found across documents:

From client_proposal.txt:
  - Project cost: $85,000
  - Annual labor savings: $240,000
  - Break-even: 4.3 months
  - 3-year ROI: 847%

No financial figures found in ai_trends_2025.txt or mcp_technical_overview.txt.

============================================================
  Lab complete. All MCP capabilities demonstrated.
============================================================
```
