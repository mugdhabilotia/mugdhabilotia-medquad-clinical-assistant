import asyncio
from app.agents.researcher_agent import get_researcher_mcp_toolset


async def main():
    toolset = get_researcher_mcp_toolset()

    # Query the MCP server for available tools
    tools = await toolset.get_tools()

    print(f"\n--- Found {len(tools)} Tool(s) in McpToolset ---")
    for tool in tools:
        name = getattr(tool, "name", str(tool))
        description = getattr(tool, "description", "No description")
        print(f"• Name: {name}")
        print(f"  Description: {description}\n")


if __name__ == "__main__":
    asyncio.run(main())