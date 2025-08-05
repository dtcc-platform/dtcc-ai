#!/usr/bin/env python3
# minimal_deepseek_client.py

import json
import argparse
import asyncio
import re
from llama_cpp import Llama
from fastmcp import Client

async def main():
    parser = argparse.ArgumentParser(description="Minimal DeepSeek MCP Client")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    llm = Llama(model_path=args.model, n_ctx=1024, verbose=False)
    print("Model loaded!")
    
    # Connect to FastMCP server
    client = Client("http://localhost:8000/mcp")
    
    async with client:
        print("Connected to MCP server!")
        
        # Main loop
        while True:
            user_input = input("\n> ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            
            # Simple pre-check for obvious cases
            user_lower = user_input.lower()
            
            if any(word in user_lower for word in ['date', 'time', 'today', 'day']):
                # Direct date request
                result = await client.call_tool("get_current_date", arguments={})
                print(f"Result: {result.data}")
                continue
            
            if any(word in user_lower for word in ['hello', 'hi', 'greet']):
                # Extract name for greeting
                words = user_input.split()
                name = "there"  # default
                for i, word in enumerate(words):
                    if word.lower() in ['hello', 'hi', 'greet', 'to'] and i + 1 < len(words):
                        name = words[i + 1]
                        break
                result = await client.call_tool("say_hello", arguments={"name": name})
                print(f"Result: {result.data}")
                continue
            
            # Only use LLM for complex cases (like build_city_mesh)
            # Check if this might be a build/mesh request
            if any(word in user_lower for word in ['build', 'mesh', 'create', 'city']) or re.search(r'\d+', user_input):
                # Extract all numbers from the input
                numbers = re.findall(r'-?\d+\.?\d*', user_input)
                numbers = [float(n) for n in numbers]
                
                if len(numbers) >= 4:
                    # Take the first 4 numbers
                    xmin, ymin, xmax, ymax = numbers[0], numbers[1], numbers[2], numbers[3]
                    
                    # Ensure min/max are correct
                    if xmin > xmax:
                        xmin, xmax = xmax, xmin
                    if ymin > ymax:
                        ymin, ymax = ymax, ymin
                    
                    # Direct call with the corrected coordinates
                    print(f"Calling build_city_mesh with bounds: ({xmin}, {ymin}, {xmax}, {ymax})")
                    result = await client.call_tool("build_city_mesh", arguments={
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax
                    })
                    print(f"Result: {result.data}")
                    continue
                
            # If we get here, we couldn't handle it with simple rules
            print(f"Could not understand: {user_input}")

if __name__ == "__main__":
    asyncio.run(main())