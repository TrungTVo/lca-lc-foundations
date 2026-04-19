from fastmcp import FastMCP

mcp = FastMCP("weather_mcp_server")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return f"It's always sunny in {location}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http", show_banner=True, port=8001, path="/mcp")