
import asyncio
import json
from llama_cpp import Llama
from fastmcp import Client

# --- LLM Configuration ---
llm = Llama(
    #insert your gguf here
    model_path="",
    verbose=False
)

# --- THE CORRECTED PROMPT ---
# The curly braces around the example JSON are now doubled to "escape" them.
PROMPT_TEMPLATE = """
You are an expert assistant that maps a user's request to a specific tool.
Your only job is to respond with a single, clean JSON object and nothing else.

Here are the available tools:
- "get_current_date": Use this tool if the user asks for the current date or time. It takes no arguments.
- "say_hello": Use this tool if the user asks for a greeting. It has one argument: "name". You can use "User" as the default name.

If no tool matches the user's request, respond with {{"tool_name": "no_tool_found"}}.

User's request: "{query}"
JSON response:
"""

def understand_intent_with_llm(query: str) -> dict | None:
    """Uses the LLM to determine the user's intent and returns it as a dictionary."""
    # This line will now work correctly because the prompt is fixed.
    prompt = PROMPT_TEMPLATE.format(query=query)
    
    try:
        output = llm(prompt, max_tokens=100, stop=["}"], echo=False)
        json_text = output["choices"][0]["text"] + "}"
        return json.loads(json_text)
    except Exception as e:
        print(f"[LLM Error: {e}]")
        return None

async def main():
    """Main loop to connect to the MCP server and use the LLM for intent."""
    client = Client("http://localhost:8000/mcp")
    print("LLM-Powered Client started. Type 'exit' to quit.")

    while True:
        user_query = input("> ")
        if user_query.lower() == 'exit':
            break

        intent = understand_intent_with_llm(user_query)

        if not isinstance(intent, dict):
            print("Sorry, the LLM did not produce a valid response.")
            continue

        tool_name = intent.get("tool_name")
        if not tool_name or tool_name == "no_tool_found":
            print("Sorry, I can't find a tool for that request.")
            continue
        tool_name = tool_name.lower()

        arguments = intent.get("arguments", {})
        
        if tool_name == "say_hello" and "name" not in arguments:
            print("[Client Logic] 'name' argument was missing. Adding default 'User'.")
            arguments["name"] = "User"

        print(f"LLM decided to call tool: {tool_name}...")
        try:
            async with client:
                result = await client.call_tool(tool_name, arguments=arguments)
            print(f"Server response: {result.data}\n")
        except Exception as e:
            print(f"ERROR during MCP call: {e}")

if __name__ == "__main__":
    asyncio.run(main())