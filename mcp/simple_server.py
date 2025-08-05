# server.py (with date tool)
from fastmcp import FastMCP
from datetime import date # 1. Import the date object

# Create an MCP server instance
mcp = FastMCP("HelloWorldServer")

@mcp.tool()
def say_hello(name: str) -> str:
    """
    A simple tool that returns a greeting.

    :param name: The name to include in the greeting.
    :return: A personalized greeting message.
    """
    print(f"Server received a request with name: {name}")
    return f"Hello, {name}!"

# 2. Add the new tool for getting the date
@mcp.tool()
def get_current_date() -> str:
    """
    Returns the current date as a string in YYYY-MM-DD format.
    :return: The current date.
    """
    print("Server received a request for the current date.")
    # The isoformat() method returns the date as 'YYYY-MM-DD'
    return date.today().isoformat()

if __name__ == "__main__":
    # The server run command stays the same
    mcp.run(transport="http", host="0.0.0.0", port=8000)