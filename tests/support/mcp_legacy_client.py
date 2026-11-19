"""Cross-version acceptance runner, executed by an isolated mcp==1.26.0 Python."""

import asyncio
import json
import os
import sys
from importlib.metadata import version

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    assert version("mcp") == "1.26.0"
    params = StdioServerParameters(
        command=sys.argv[1], args=[sys.argv[2]], env=dict(os.environ)
    )
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as client,
    ):
        initialized = await client.initialize()
        catalog = await client.list_tools()
        assert [tool.name for tool in catalog.tools] == [
            "execute_code",
            "get_status",
            "reset_browser",
        ]
        first = await client.call_tool("execute_code", {"code": "answer = 42"})
        assert not first.isError
        second = await client.call_tool("execute_code", {"code": "print(answer)"})
        assert not second.isError and "42" in second.content[0].text
        status = await client.call_tool("get_status", {})
        assert not status.isError
        reset = await client.call_tool("reset_browser", {"clear_variables": True})
        assert not reset.isError and json.loads(reset.content[0].text)["ok"]
        sys.stdout.write(
            json.dumps(
                {
                    "sdk": version("mcp"),
                    "protocol": initialized.protocolVersion,
                    "state": 42,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    asyncio.run(main())
