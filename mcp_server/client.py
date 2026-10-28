"""Thin MCP client used by the application (Streamlit) to call save_draft_mcp."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent


async def _call(arguments: dict) -> dict:
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"],
                                   env=dict(os.environ), cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("save_draft_mcp", arguments)
    return json.loads(result.content[0].text)


def call_save_draft(arguments: dict) -> dict:
    return asyncio.run(_call(arguments))


async def _list_tools() -> list[str]:
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"],
                                   env=dict(os.environ), cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return [tool.name for tool in (await session.list_tools()).tools]


def list_server_tools() -> list[str]:
    return asyncio.run(_list_tools())