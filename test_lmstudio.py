#!/usr/bin/env python3
# test_lmstudio.py

import asyncio
import httpx
import json
import argparse
from typing import Dict, Any

async def query_lm_studio(
    prompt: str, 
    base_url: str = "http://localhost:1234",
    temperature: float = 0.7,
    max_tokens: int = 500,
    system_prompt: str = None
) -> Dict[str, Any]:
    """Query LM Studio's OpenAI-compatible API"""
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "content": data['choices'][0]['message']['content'],
                    "model": data.get('model', 'unknown'),
                    "usage": data.get('usage', {})
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except httpx.ConnectError:
            return {
                "success": False,
                "error": "Cannot connect to LM Studio. Make sure it's running and the server is started."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error: {str(e)}"
            }

async def test_function_calling():
    """Test function calling prompts"""
    print("\n" + "="*60)
    print("FUNCTION CALLING TESTS")
    print("="*60)
    
    test_cases = [
        {
            "name": "Simple greeting",
            "prompt": """Convert to JSON function call:
Request: say hello to john

Functions:
- say_hello(name) - greets someone
- get_current_date() - gets date

Output only valid JSON: {"tool": "function_name", "params": {}}""",
            "expected": '{"tool": "say_hello", "params": {"name": "john"}}'
        },
        {
            "name": "Date request",
            "prompt": """Convert to JSON function call:
Request: what is the date today

Functions:
- say_hello(name) - greets someone  
- get_current_date() - gets date

Output only valid JSON: {"tool": "function_name", "params": {}}""",
            "expected": '{"tool": "get_current_date", "params": {}}'
        },
        {
            "name": "Complex coordinates",
            "prompt": """Convert to JSON function call:
Request: build city mesh for area 320000 6400000 322000 6402000

Functions:
- build_city_mesh(xmin, ymin, xmax, ymax) - builds city mesh

Output only valid JSON with all 4 coordinates.""",
            "expected": '{"tool": "build_city_mesh", "params": {"xmin": 320000, "ymin": 6400000, "xmax": 322000, "ymax": 6402000}}'
        }
    ]
    
    for test in test_cases:
        print(f"\n### Test: {test['name']}")
        print(f"Expected: {test['expected']}")
        
        result = await query_lm_studio(test['prompt'], temperature=0.1, max_tokens=100)
        
        if result['success']:
            print(f"Response: {result['content']}")
            
            # Try to validate JSON
            try:
                parsed = json.loads(result['content'])
                print("✓ Valid JSON")
            except:
                print("✗ Invalid JSON")
        else:
            print(f"Error: {result['error']}")

async def interactive_mode(base_url: str):
    """Interactive prompt testing"""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Type 'quit' to exit, 'help' for options\n")
    
    temperature = 0.7
    max_tokens = 500
    system_prompt = "You are a helpful assistant."
    
    while True:
        prompt = input("\n> ").strip()
        
        if prompt.lower() == 'quit':
            break
            
        if prompt.lower() == 'help':
            print("""
Commands:
  quit - Exit the program
  help - Show this help
  temp <value> - Set temperature (0.0-2.0)
  tokens <value> - Set max tokens
  system <prompt> - Set system prompt
  clear - Clear system prompt
            """)
            continue
            
        if prompt.startswith('temp '):
            try:
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}")
            except:
                print("Invalid temperature value")
            continue
            
        if prompt.startswith('tokens '):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"Max tokens set to {max_tokens}")
            except:
                print("Invalid token value")
            continue
            
        if prompt.startswith('system '):
            system_prompt = prompt[7:]
            print(f"System prompt set to: {system_prompt}")
            continue
            
        if prompt == 'clear':
            system_prompt = None
            print("System prompt cleared")
            continue
        
        if not prompt:
            continue
        
        print("\nQuerying LM Studio...")
        result = await query_lm_studio(
            prompt, 
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )
        
        if result['success']:
            print(f"\n--- Response ---")
            print(result['content'])
            print(f"\n--- Model: {result['model']} ---")
            if result['usage']:
                print(f"Tokens - Prompt: {result['usage'].get('prompt_tokens', 0)}, "
                      f"Completion: {result['usage'].get('completion_tokens', 0)}, "
                      f"Total: {result['usage'].get('total_tokens', 0)}")
        else:
            print(f"\nError: {result['error']}")

async def batch_test(base_url: str, test_file: str):
    """Run batch tests from a file"""
    print("\n" + "="*60)
    print(f"BATCH TEST: {test_file}")
    print("="*60)
    
    try:
        with open(test_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Test file '{test_file}' not found")
        return
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n### Test {i}/{len(prompts)}")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        
        result = await query_lm_studio(prompt, base_url=base_url, temperature=0.1)
        
        if result['success']:
            print(f"Response: {result['content'][:200]}..." if len(result['content']) > 200 else f"Response: {result['content']}")
        else:
            print(f"Error: {result['error']}")

async def main():
    parser = argparse.ArgumentParser(description="Test LM Studio API")
    parser.add_argument("--url", default="http://localhost:1234", help="LM Studio API URL")
    parser.add_argument("--mode", choices=['interactive', 'function', 'batch'], default='interactive',
                        help="Test mode: interactive, function, or batch")
    parser.add_argument("--file", help="Test file for batch mode")
    
    args = parser.parse_args()
    
    # Test connection
    print(f"Testing connection to LM Studio at {args.url}...")
    test_result = await query_lm_studio("Hello", base_url=args.url)
    
    if test_result['success']:
        print(f"✓ Connected! Model: {test_result.get('model', 'unknown')}")
    else:
        print(f"✗ {test_result['error']}")
        return
    
    # Run selected mode
    if args.mode == 'interactive':
        await interactive_mode(args.url)
    elif args.mode == 'function':
        await test_function_calling()
    elif args.mode == 'batch':
        if not args.file:
            print("Error: --file required for batch mode")
            return
        await batch_test(args.url, args.file)

if __name__ == "__main__":
    asyncio.run(main())