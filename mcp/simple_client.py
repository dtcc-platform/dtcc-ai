# client.py (calling both tools)
import asyncio
from fastmcp import Client

async def main():
    client = Client("http://localhost:8000/mcp")

    try:
        async with client:
            await client.ping()
            print("Successfully connected to the server.")

            # The list of tools will now include 'get_current_date'
            list_of_tools = await client.list_tools()
            print("Available tools:", [tool.name for tool in list_of_tools])

            # --- Call the 'say_hello' tool (as before) ---
            if any(tool.name == "say_hello" for tool in list_of_tools):
                hello_result = await client.call_tool(
                    "say_hello",
                    arguments={"name": "World"}
                )
                print(f"Response from hello tool: {hello_result.data}")
            else:
                print("Tool 'say_hello' not found on the server.")

            # --- Call the new 'get_current_date' tool ---
            if any(tool.name == "get_current_date" for tool in list_of_tools):
                # This tool has no parameters, so we pass an empty dictionary
                date_result = await client.call_tool(
                    "get_current_date",
                    arguments={}
                )
                print(f"Response from date tool: {date_result.data}")
            else:
                print("Tool 'get_current_date' not found on the server.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Please ensure the server is running and accessible at http://localhost:8000")

if __name__ == "__main__":
    asyncio.run(main())