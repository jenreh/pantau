from __future__ import annotations

import asyncio

from pantau.home_mcp.server import mcp


async def _list_components() -> tuple[set[str], set[str]]:
    tools = await mcp.list_tools(run_middleware=False)
    resources = await mcp.list_resources()
    return {tool.name for tool in tools}, {str(resource.uri) for resource in resources}


def test_home_mcp_composes_legacy_and_native_servers() -> None:
    tool_names, resource_uris = asyncio.run(_list_components())

    assert "harmony_get_status" in tool_names
    assert "hue_get_bridge_info" in tool_names
    assert "harmony://status" in resource_uris
