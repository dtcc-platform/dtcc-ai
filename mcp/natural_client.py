# natural_language_client.py
import asyncio
from fastmcp import Client

# This is our "Natural Language Understanding" function.
# It's a bare-minimum implementation using keyword checking.
def understand_intent(query: str) -> tuple[str, dict] | None:
    """
    Analyzes the user's query to determine which tool to call and with what arguments.

    :param query: The user's input string.
    :return: A tuple containing (tool_name, arguments_dict), or None if not understood.
    """
    query = query.lower() # Convert to lowercase for easier matching

    if "date" in query or "time" in query:
        # If the user asks for the date or time, map to the 'get_current_date' tool.
        return ("get_current_date", {})

    if "hello" in query or "greet" in query:
        # If the user wants a greeting, map to the 'say_hello' tool.
        # For this bare-minimum example, we'll just hardcode the name.
        return ("say_hello", {"name": "User"})

    # If no keywords match, we don't know what to do.
    return None


async def main():
    """
    Main loop to connect to the MCP server, take user input, and execute tools.
    """
    client = Client("http://localhost:8000/mcp")
    print("Natural Language Client started. Type 'exit' to quit.")

    try:
        async with client:
            while True:
                # 1. Get input from the user
                user_query = input("> ")
                if user_query.lower() == 'exit':
                    break

                # 2. Understand the user's intent
                intent = understand_intent(user_query)

                if not intent:
                    print("Sorry, I don't understand that.")
                    continue

                tool_name, arguments = intent

                # 3. Call the appropriate tool on the MCP server
                print(f"Understood! Calling tool: {tool_name}...")
                result = await client.call_tool(tool_name, arguments=arguments)

                # 4. Print the result
                print(f"Server response: {result.data}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())