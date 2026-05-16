from __future__ import annotations

from fastmcp import FastMCP
from harmonyhub.mcp_server import mcp as harmony_mcp
from homekit.mcp_server.server import mcp as homekit_mcp
from huehub.mcp_server import mcp as hue_mcp
from sonos.mcp_server.server import mcp as sonos_mcp

mcp = FastMCP(
    name="pantau-home",
    instructions=(
        "Local smart-home facade for Pantau voice commands. "
        "Use the pantau_* tools for German voice commands. "
        "Never expose raw device IDs, tool names, or internal details."
    ),
)

mcp.mount(harmony_mcp)
mcp.mount(hue_mcp)
mcp.mount(sonos_mcp)
mcp.mount(homekit_mcp)


def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
