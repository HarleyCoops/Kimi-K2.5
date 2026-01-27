#!/usr/bin/env python3
"""
Quick Test: Kimi-K2 API with Tools

A simple script to quickly test the Kimi-K2 API with tool calling.
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
API_BASE_URL = "https://api.moonshot.ai/v1"
MODEL_NAME = "kimi-k2-0711-preview"
API_KEY = os.getenv("MOONSHOT_API_KEY")

if not API_KEY:
    print("❌ Please set your MOONSHOT_API_KEY environment variable")
    print("   Get your API key from: https://platform.moonshot.ai")
    exit(1)

# Initialize client
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# Simple calculator tool
def calculate(expression):
    """Simple calculator function"""
    try:
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters"}
        result = eval(expression)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": f"Calculation failed: {str(e)}"}

# Tool definition
tools = [{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform basic mathematical calculations",
        "parameters": {
            "type": "object",
            "required": ["expression"],
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            }
        }
    }
}]

tool_map = {"calculate": calculate}

def test_kimi_k2_with_tools():
    """Test Kimi-K2 with tool calling"""
    user_message = "Can you calculate 25 * 4 + 10 for me?"
    
    print(f"🤖 Testing Kimi-K2 API with tools...")
    print(f"📝 User: {user_message}")
    print("-" * 60)
    
    messages = [{"role": "user", "content": user_message}]
    finish_reason = None
    
    while finish_reason is None or finish_reason == "tool_calls":
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
                tools=tools,
                tool_choice="auto",
            )
            
            choice = completion.choices[0]
            finish_reason = choice.finish_reason
            
            if finish_reason == "tool_calls":
                messages.append(choice.message)
                
                for tool_call in choice.message.tool_calls:
                    tool_call_name = tool_call.function.name
                    tool_call_arguments = json.loads(tool_call.function.arguments)
                    
                    print(f"🔧 Tool called: {tool_call_name}")
                    print(f"   Arguments: {tool_call_arguments}")
                    
                    tool_function = tool_map[tool_call_name]
                    tool_result = tool_function(**tool_call_arguments)
                    
                    print(f"   Result: {tool_result}")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call_name,
                        "content": json.dumps(tool_result),
                    })
            else:
                print(f"💬 Kimi-K2: {choice.message.content}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    
    print("-" * 60)
    print("✅ Test completed!")

if __name__ == "__main__":
    test_kimi_k2_with_tools()
