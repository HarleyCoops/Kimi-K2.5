#!/usr/bin/env python3
"""
Kimi-K2 API Example with Tools Enabled

This script demonstrates how to use the Kimi-K2 model API with tool calling capabilities.
The Kimi-K2 API is available at https://platform.moonshot.ai

Requirements:
- openai>=1.0.0
- requests (for manual parsing example)
"""

import json
import os
from openai import OpenAI
import requests
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
API_BASE_URL = "https://api.moonshot.ai/v1"  # Kimi-K2 API endpoint
MODEL_NAME = "moonshotai/Kimi-K2-Instruct"

# You'll need to set your API key as an environment variable
# export MOONSHOT_API_KEY="your_api_key_here"
API_KEY = os.getenv("MOONSHOT_API_KEY")

if not API_KEY:
    print("⚠️  Please set your MOONSHOT_API_KEY environment variable")
    print("   You can get an API key from https://platform.moonshot.ai")
    exit(1)

# Initialize the client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# Example tool: Weather function
def get_weather(city):
    """Mock weather function - in a real scenario, this would call a weather API"""
    weather_data = {
        "Beijing": {"weather": "Sunny", "temperature": "22°C", "humidity": "45%"},
        "Shanghai": {"weather": "Cloudy", "temperature": "18°C", "humidity": "60%"},
        "New York": {"weather": "Rainy", "temperature": "15°C", "humidity": "80%"},
        "London": {"weather": "Foggy", "temperature": "12°C", "humidity": "70%"}
    }
    return weather_data.get(city, {"weather": "Unknown", "temperature": "N/A", "humidity": "N/A"})

# Example tool: Calculator function
def calculate(expression):
    """Simple calculator function"""
    try:
        # Only allow basic arithmetic for safety
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression"}
        
        result = eval(expression)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": f"Calculation failed: {str(e)}"}

# Tool definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a specific city. Call this tool when the user asks about weather conditions.",
            "parameters": {
                "type": "object",
                "required": ["city"],
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get weather for"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic mathematical calculations. Call this tool when the user asks for calculations or math problems.",
            "parameters": {
                "type": "object",
                "required": ["expression"],
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 2 * 3')"
                    }
                }
            }
        }
    }
]

# Tool mapping for easy calling
tool_map = {
    "get_weather": get_weather,
    "calculate": calculate
}

def chat_with_tools_non_streaming(user_message, temperature=0.3):
    """
    Chat with Kimi-K2 using tools in non-streaming mode
    """
    print(f"🤖 User: {user_message}")
    print("-" * 80)
    
    messages = [{"role": "user", "content": user_message}]
    finish_reason = None
    
    while finish_reason is None or finish_reason == "tool_calls":
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
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
                    
                    print(f"🔧 Calling tool: {tool_call_name}")
                    print(f"   Arguments: {tool_call_arguments}")
                    
                    tool_function = tool_map[tool_call_name]
                    tool_result = tool_function(**tool_call_arguments)
                    
                    print(f"   Result: {tool_result}")
                    print()
                    
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
    
    print("-" * 80)
    return choice.message.content if finish_reason != "tool_calls" else None

def chat_with_tools_streaming(user_message, temperature=0.3):
    """
    Chat with Kimi-K2 using tools in streaming mode
    """
    print(f"🤖 User: {user_message}")
    print("-" * 80)
    
    messages = [{"role": "user", "content": user_message}]
    finish_reason = None
    msg = ''
    
    while finish_reason is None or finish_reason == "tool_calls":
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice="auto",
                stream=True
            )
            
            tool_calls = []
            for chunk in completion:
                delta = chunk.choices[0].delta
                
                if delta.content:
                    msg += delta.content
                    print(delta.content, end='', flush=True)
                
                if delta.tool_calls:
                    for tool_call_chunk in delta.tool_calls:
                        if tool_call_chunk.index is not None:
                            # Extend the tool_calls list
                            while len(tool_calls) <= tool_call_chunk.index:
                                tool_calls.append({
                                    "id": "",
                                    "type": "function",
                                    "function": {
                                        "name": "",
                                        "arguments": ""
                                    }
                                })
                            
                            tc = tool_calls[tool_call_chunk.index]
                            
                            if tool_call_chunk.id:
                                tc["id"] += tool_call_chunk.id
                            if tool_call_chunk.function.name:
                                tc["function"]["name"] += tool_call_chunk.function.name
                            if tool_call_chunk.function.arguments:
                                tc["function"]["arguments"] += tool_call_chunk.function.arguments
                
                finish_reason = chunk.choices[0].finish_reason
            
            if finish_reason == "tool_calls":
                # Append the assistant message with tool calls to messages list
                assistant_message = {
                    "role": "assistant",
                    "content": msg if msg else None,
                    "tool_calls": [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    } for tc in tool_calls]
                }
                messages.append(assistant_message)
                
                print("\n🔧 Tool calls detected:")
                for tool_call in tool_calls:
                    tool_call_name = tool_call['function']['name']
                    tool_call_arguments = json.loads(tool_call['function']['arguments'])
                    
                    print(f"   Calling: {tool_call_name}")
                    print(f"   Arguments: {tool_call_arguments}")
                    
                    tool_function = tool_map[tool_call_name]
                    tool_result = tool_function(**tool_call_arguments)
                    
                    print(f"   Result: {tool_result}")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "name": tool_call_name,
                        "content": json.dumps(tool_result),
                    })
                
                # Reset message for next iteration
                msg = ''
                print()
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break
    
    print("\n" + "-" * 80)
    return msg

def extract_tool_call_info(tool_call_rsp: str):
    """
    Manually parse tool calls from Kimi-K2 output
    """
    if '<|tool_calls_section_begin|>' not in tool_call_rsp:
        return []
    
    pattern = r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>"
    tool_calls_sections = re.findall(pattern, tool_call_rsp, re.DOTALL)
    
    func_call_pattern = r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*<\|tool_call_end\|>"
    tool_calls = []
    
    for match in re.findall(func_call_pattern, tool_calls_sections[0], re.DOTALL):
        function_id, function_args = match
        function_name = function_id.split('.')[1].split(':')[0]
        tool_calls.append({
            "id": function_id,
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": function_args
            }
        })
    
    return tool_calls

def manual_tool_parsing_example(user_message):
    """
    Example of manually parsing tool calls (useful when service doesn't provide tool-call parser)
    """
    print(f"🤖 User: {user_message}")
    print("-" * 80)
    
    messages = [{"role": "user", "content": user_message}]
    
    try:
        # This would require the transformers library and model tokenizer
        # For this example, we'll show the structure
        print("📝 Manual parsing example structure:")
        print("   - Use transformers.AutoTokenizer to tokenize messages")
        print("   - Send request to completions endpoint")
        print("   - Parse tool calls using extract_tool_call_info()")
        print("   - Execute tools and continue conversation")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """
    Main function demonstrating different ways to use Kimi-K2 with tools
    """
    print("🚀 Kimi-K2 API Example with Tools Enabled")
    print("=" * 80)
    
    # Example 1: Non-streaming mode with weather tool
    print("\n📋 Example 1: Non-streaming mode with weather tool")
    chat_with_tools_non_streaming("What's the weather like in Beijing today?")
    
    # Example 2: Non-streaming mode with calculator tool
    print("\n📋 Example 2: Non-streaming mode with calculator tool")
    chat_with_tools_non_streaming("Can you calculate 15 * 23 + 7?")
    
    # Example 3: Streaming mode with multiple tools
    print("\n📋 Example 3: Streaming mode with multiple tools")
    chat_with_tools_streaming("What's the weather in Shanghai and can you calculate 100 / 4?")
    
    # Example 4: Manual parsing structure
    print("\n📋 Example 4: Manual parsing structure")
    manual_tool_parsing_example("What's 2 + 2?")
    
    print("\n✅ Examples completed!")
    print("\n💡 Tips:")
    print("   - Set MOONSHOT_API_KEY environment variable")
    print("   - Kimi-K2 supports both OpenAI and Anthropic-compatible APIs")
    print("   - Temperature mapping: real_temperature = request_temperature * 0.6")
    print("   - Tool calling requires specific parser support in inference engines")

if __name__ == "__main__":
    main() 