"""Local MCP server exposing the ONLY write tool: save_draft_mcp (Developer B, task B2).

Run:  uv run python -m mcp_server.server
Deliberately exposes no token-issuing tool and no export tool.
"""
from __future__ import annotations

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

try:  # mcp >= 2 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from mcp_server.service import save_approved_draft

mcp = _Server("smartinbox-drafts")


@mcp.tool()
def save_draft_mcp(action_id: str, email_id: str, primary_department: str,
                   secondary_department: str | None, draft_subject: str,
                   draft_body: str, approval_token: str) -> dict:
    """Save a human-approved draft and department assignment locally.

    Requires a server-issued approval_token bound to the exact payload. Never call
    this tool on your own initiative: only the application calls it after approval.
    """
    return save_approved_draft({
        "action_id": action_id, "email_id": email_id,
        "primary_department": primary_department, "secondary_department": secondary_department,
        "draft_subject": draft_subject, "draft_body": draft_body, "approval_token": approval_token,
    })


if __name__ == "__main__":
    mcp.run()