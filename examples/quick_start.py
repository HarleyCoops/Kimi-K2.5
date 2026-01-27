#!/usr/bin/env python3
"""
Kimi K2.5 Quick Start

Get started with Kimi K2.5 in minutes!

This example covers:
1. Basic chat completion
2. Thinking mode with reasoning traces
3. Instant mode for fast responses
4. Simple tool calling

Prerequisites:
    pip install -r requirements.txt
    export MOONSHOT_API_KEY="your_api_key"

Usage:
    python examples/quick_start.py
"""

import sys
sys.path.insert(0, '.')

from kimi_client import KimiClient, KimiMode


def demo_basic_chat():
    """Basic chat completion"""
    print("\n" + "="*60)
    print("BASIC CHAT")
    print("="*60)

    client = KimiClient()

    # Simple question
    response = client.chat(
        "What are the three laws of robotics?",
        mode=KimiMode.INSTANT  # Fast response
    )

    print(f"\nQuestion: What are the three laws of robotics?")
    print(f"\nResponse:\n{response.content}")
    print(f"\nTokens used: {response.total_tokens}")


def demo_thinking_mode():
    """Thinking mode with reasoning traces"""
    print("\n" + "="*60)
    print("THINKING MODE")
    print("="*60)

    client = KimiClient(default_mode=KimiMode.THINKING)

    # Complex reasoning question
    response = client.chat(
        "Which is larger: 9.11 or 9.9? Think carefully and explain your reasoning."
    )

    print(f"\nQuestion: Which is larger: 9.11 or 9.9?")

    if response.reasoning:
        print(f"\n[Reasoning Trace]")
        print("-" * 40)
        print(response.reasoning[:500])
        if len(response.reasoning) > 500:
            print("...")

    print(f"\n[Final Answer]")
    print("-" * 40)
    print(response.content)


def demo_tool_calling():
    """Simple tool calling example"""
    print("\n" + "="*60)
    print("TOOL CALLING")
    print("="*60)

    client = KimiClient(default_mode=KimiMode.AGENT)

    # Define a simple calculator tool
    tools = [{
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "required": ["expression"],
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression (e.g., '2 + 2 * 3')"
                    }
                }
            }
        }
    }]

    def calculate(expression: str) -> dict:
        """Safe calculator"""
        try:
            # Only allow safe characters
            allowed = set('0123456789+-*/(). ')
            if all(c in allowed for c in expression):
                result = eval(expression)
                return {"result": result}
            return {"error": "Invalid expression"}
        except Exception as e:
            return {"error": str(e)}

    tool_map = {"calculate": calculate}

    # Execute with tools
    response = client.execute_with_tools(
        "What is 15 * 23 + 7?",
        tools=tools,
        tool_map=tool_map,
        on_tool_call=lambda name, args: print(f"  -> Calling {name} with {args}"),
        on_tool_result=lambda name, result: print(f"  <- Result: {result}"),
    )

    print(f"\nQuestion: What is 15 * 23 + 7?")
    print(f"\nTool Execution:")
    print(f"\nFinal Answer: {response.content}")


def demo_streaming():
    """Streaming responses"""
    print("\n" + "="*60)
    print("STREAMING")
    print("="*60)

    client = KimiClient()

    print("\nStreaming response for: 'Write a haiku about AI'\n")

    # Stream the response
    for chunk in client.stream("Write a haiku about artificial intelligence"):
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print("\n")


def main():
    """Run all demos"""
    print("=" * 60)
    print("KIMI K2.5 QUICK START")
    print("=" * 60)
    print("\nThis demo showcases the core Kimi K2.5 capabilities.")
    print("Make sure MOONSHOT_API_KEY is set in your environment.\n")

    try:
        # Check API key
        from config import validate_api_key
        if not validate_api_key():
            print("ERROR: MOONSHOT_API_KEY not set!")
            print("Set it with: export MOONSHOT_API_KEY='your_key'")
            print("Get a key from: https://platform.moonshot.ai")
            return

        # Run demos
        demo_basic_chat()
        demo_thinking_mode()
        demo_tool_calling()
        demo_streaming()

        print("\n" + "="*60)
        print("QUICK START COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("  - Try multimodal: python examples/multimodal_demo.py")
        print("  - Explore swarm: python examples/swarm_demo.py")
        print("  - Extended tools: python examples/tool_calling_demo.py")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure your API key is valid and you have credits.")


if __name__ == "__main__":
    main()
