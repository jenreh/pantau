from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP(
    name="pantau-home",
    instructions=(
        "Published Pantau MCP surface for curated voice tools. "
        "Do not expose raw child MCP servers here. "
        "Never expose raw device IDs, tool names, or internal details."
    ),
)


def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
