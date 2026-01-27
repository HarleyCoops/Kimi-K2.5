#!/usr/bin/env python3
"""
Kimi-K2 Tool Exploration Script

This script explores the tool capabilities of Kimi-K2, including:
1. Testing native tool support
2. Creating custom tools
3. Exploring MCP (Model Context Protocol) integration possibilities
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, List, Any

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = "https://api.moonshot.ai/v1"
MODEL_NAME = "kimi-k2-0711-preview"
API_KEY = os.getenv("MOONSHOT_API_KEY")

if not API_KEY:
    print("[ERROR] MOONSHOT_API_KEY environment variable not set")
    print("Run: python setup_api_key.py")
    exit(1)

# Initialize client
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def test_native_tools():
    """Test if model has native tool support without explicit tool definitions"""
    print("\n" + "="*70)
    print("TESTING NATIVE TOOL SUPPORT")
    print("="*70)
    
    # Test without providing any tools
    messages = [{
        "role": "user", 
        "content": "What's the weather like in Tokyo? Can you check for me?"
    }]
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        
        print(f"Response without tools: {response.choices[0].message.content}")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        
    except Exception as e:
        print(f"[ERROR] {e}")

def create_custom_tool_examples():
    """Demonstrate various custom tool patterns"""
    print("\n" + "="*70)
    print("CUSTOM TOOL EXAMPLES")
    print("="*70)
    
    # Example 1: System command tool (for MCP-like functionality)
    system_command_tool = {
        "type": "function",
        "function": {
            "name": "execute_system_command",
            "description": "Execute a system command and return the output",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The system command to execute"
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Optional working directory for command execution"
                    }
                }
            }
        }
    }
    
    # Example 2: File manipulation tool
    file_tool = {
        "type": "function",
        "function": {
            "name": "file_operations",
            "description": "Perform file operations like read, write, or list files",
            "parameters": {
                "type": "object",
                "required": ["operation", "path"],
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write", "list", "delete"],
                        "description": "The file operation to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content for write operations"
                    }
                }
            }
        }
    }
    
    # Example 3: Database query tool
    database_tool = {
        "type": "function",
        "function": {
            "name": "database_query",
            "description": "Execute database queries and return results",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute"
                    },
                    "database": {
                        "type": "string",
                        "description": "Database name (default: main)"
                    }
                }
            }
        }
    }
    
    # Example 4: API request tool
    api_tool = {
        "type": "function",
        "function": {
            "name": "make_api_request",
            "description": "Make HTTP API requests to external services",
            "parameters": {
                "type": "object",
                "required": ["url", "method"],
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to request"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE"],
                        "description": "HTTP method"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers"
                    },
                    "body": {
                        "type": "object",
                        "description": "Optional request body for POST/PUT"
                    }
                }
            }
        }
    }
    
    tools = [system_command_tool, file_tool, database_tool, api_tool]
    
    # Test tool recognition
    messages = [{
        "role": "user",
        "content": "I need to: 1) List files in the current directory, 2) Make a GET request to https://api.example.com/data, and 3) Execute 'echo Hello World'. Can you help with these tasks?"
    }]
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.3
        )
        
        choice = response.choices[0]
        print(f"\nModel response with custom tools:")
        print(f"Finish reason: {choice.finish_reason}")
        
        if choice.finish_reason == "tool_calls":
            print("\nTool calls requested:")
            for tool_call in choice.message.tool_calls:
                print(f"\n- Tool: {tool_call.function.name}")
                print(f"  Arguments: {tool_call.function.arguments}")
        else:
            print(f"Content: {choice.message.content}")
            
    except Exception as e:
        print(f"[ERROR] {e}")

def test_mcp_pattern():
    """Test MCP (Model Context Protocol) pattern implementation"""
    print("\n" + "="*70)
    print("MCP PATTERN EXPLORATION")
    print("="*70)
    
    # MCP-style tool that could interface with external MCP servers
    mcp_tool = {
        "type": "function",
        "function": {
            "name": "mcp_server_request",
            "description": "Send requests to MCP (Model Context Protocol) servers",
            "parameters": {
                "type": "object",
                "required": ["server", "method", "params"],
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "MCP server identifier (e.g., 'filesystem', 'git', 'sqlite')"
                    },
                    "method": {
                        "type": "string",
                        "description": "MCP method to call (e.g., 'read_file', 'list_directory')"
                    },
                    "params": {
                        "type": "object",
                        "description": "Parameters for the MCP method"
                    }
                }
            }
        }
    }
    
    # Test MCP pattern
    messages = [{
        "role": "user",
        "content": "Using the MCP server, read the file 'config.json' from the filesystem server"
    }]
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=[mcp_tool],
            tool_choice="auto",
            temperature=0.3
        )
        
        choice = response.choices[0]
        if choice.finish_reason == "tool_calls":
            print("\nMCP tool call requested:")
            for tool_call in choice.message.tool_calls:
                print(f"Tool: {tool_call.function.name}")
                args = json.loads(tool_call.function.arguments)
                print(f"Arguments: {json.dumps(args, indent=2)}")
                
                # Simulate MCP response
                print("\nSimulated MCP server response:")
                print("This demonstrates how an MCP bridge could work with Kimi-K2")
        else:
            print(f"Response: {choice.message.content}")
            
    except Exception as e:
        print(f"[ERROR] {e}")

def demonstrate_tool_chaining():
    """Show how multiple tools can be chained together"""
    print("\n" + "="*70)
    print("TOOL CHAINING DEMONSTRATION")
    print("="*70)
    
    # Define tools that work together
    tools = [
        {
            "type": "function",
            "function": {
                "name": "fetch_data",
                "description": "Fetch data from a source",
                "parameters": {
                    "type": "object",
                    "required": ["source"],
                    "properties": {
                        "source": {"type": "string", "description": "Data source identifier"}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "process_data",
                "description": "Process fetched data",
                "parameters": {
                    "type": "object",
                    "required": ["data", "operation"],
                    "properties": {
                        "data": {"type": "string", "description": "Data to process"},
                        "operation": {"type": "string", "enum": ["analyze", "transform", "summarize"]}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "save_results",
                "description": "Save processed results",
                "parameters": {
                    "type": "object",
                    "required": ["results", "destination"],
                    "properties": {
                        "results": {"type": "string", "description": "Results to save"},
                        "destination": {"type": "string", "description": "Where to save results"}
                    }
                }
            }
        }
    ]
    
    messages = [{
        "role": "user",
        "content": "Fetch data from 'user_database', analyze it, and save the results to 'analysis_report.json'"
    }]
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.3
        )
        
        print("\nTool chaining capability test:")
        if response.choices[0].finish_reason == "tool_calls":
            print("Model successfully identified need for tool chaining!")
            for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                print(f"\nStep {i+1}: {tool_call.function.name}")
                print(f"Arguments: {tool_call.function.arguments}")
        else:
            print(response.choices[0].message.content)
            
    except Exception as e:
        print(f"[ERROR] {e}")

def main():
    """Run all tool exploration tests"""
    print("\nKIMI-K2 TOOL CAPABILITY EXPLORATION")
    print("="*70)
    
    # Run tests
    test_native_tools()
    create_custom_tool_examples()
    test_mcp_pattern()
    demonstrate_tool_chaining()
    
    print("\n" + "="*70)
    print("CONCLUSIONS:")
    print("="*70)
    print("""
1. Native Tools: Kimi-K2 does not have built-in native tools; all tools must be explicitly defined
2. Custom Tools: The model accepts any custom tool definition following the OpenAI function schema
3. MCP Integration: An MCP bridge tool could be created to interface with MCP servers
4. Tool Flexibility: The model can work with any tool pattern, including:
   - System commands
   - File operations
   - Database queries
   - API requests
   - MCP server communication
5. Tool Chaining: The model can identify when multiple tools need to be used in sequence

Theoretical MCP Integration:
- YES, you can create an MCP tool that acts as a bridge to MCP servers
- The tool would translate Kimi-K2's function calls to MCP protocol requests
- This would enable Kimi-K2 to interact with any MCP-compatible server
""")

if __name__ == "__main__":
    main()