"""
Custom MCP Resource Server
Ironhack AI Consulting Course

Exposes the test_documents/ directory as proper MCP resources (not tools).
Resources are static, read-only data streams — the agent receives their
content upfront as context rather than calling them on-demand mid-reasoning.

Run via stdio (launched automatically by MultiServerMCPClient):
    python mcp_resource_server.py
"""

import asyncio
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

logging.basicConfig(level=logging.ERROR)

DOCS_DIR = Path(__file__).parent / "test_documents"

# Each document gets a URI in the form: resource://docs/<filename>
def _make_uri(filename: str) -> str:
    """Build an MCP resource URI for a document filename using this server's custom scheme.

    Example: "client_proposal.txt" → "resource://docs/client_proposal.txt"
    """
    return f"resource://docs/{filename}"

def _filename_from_uri(uri: str) -> str:
    """Strip the resource://docs/ prefix and return the bare filename for filesystem lookup."""
    return uri.replace("resource://docs/", "")

server = Server("mcp-resource-server")  # name is broadcast to the client during the MCP handshake


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """This server exposes resources only — no tools."""
    return []


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """Register every .txt file in test_documents/ as an MCP resource."""
    resources = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        resources.append(types.Resource(
            uri=_make_uri(path.name),                          # type: ignore[arg-type]
            name=path.stem.replace("_", " ").title(),
            description=f"Consulting document: {path.name}",
            mimeType="text/plain",
        ))
    return resources


@server.read_resource()
async def read_resource(uri: types.AnyUrl) -> str:
    """Return the full text content of the requested resource.

    Raises FileNotFoundError if the URI maps to a file that no longer exists;
    the MCP server layer converts this into a protocol-level error response.
    """
    filename = _filename_from_uri(str(uri))
    path = DOCS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Resource not found: {uri}")
    return path.read_text(encoding="utf-8")


async def main() -> None:
    """Start the resource server on stdio and serve until the client disconnects."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
