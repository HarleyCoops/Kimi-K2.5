#!/usr/bin/env python3
"""
Test ModelScope MCP Server Connection
"""

import os
import json
import subprocess
import sys
import time

def test_modelscope_mcp():
    """Test ModelScope MCP server with API token"""

    api_token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not api_token:
        raise SystemExit(
            "MODELSCOPE_API_TOKEN is not set. "
            "Set it in your environment before running this test."
        )
    
    print("=" * 60)
    print("Testing ModelScope MCP Server")
    print("=" * 60)
    
    # Start the MCP server process
    env = {"MODELSCOPE_API_TOKEN": api_token}
    
    try:
        process = subprocess.Popen(
            ["uvx", "modelscope-mcp-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**env, "PATH": subprocess.os.environ.get("PATH", "")}
        )
        
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        # Send the request
        request_line = json.dumps(init_request) + "\n"
        process.stdin.write(request_line)
        process.stdin.flush()
        
        # Read response
        time.sleep(1)
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line)
            print("\n[OK] MCP Server responded to initialize")
            print(f"  Response: {json.dumps(response, indent=2)[:200]}...")
        else:
            print("\n[ERROR] No response from server")
            
        # Send initialized notification
        initialized = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        request_line = json.dumps(initialized) + "\n"
        process.stdin.write(request_line)
        process.stdin.flush()
        
        # Send tools/list request
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        request_line = json.dumps(tools_request) + "\n"
        process.stdin.write(request_line)
        process.stdin.flush()
        
        time.sleep(1)
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line)
            print("\n[OK] MCP Server responded to tools/list")
            
            if "result" in response and "tools" in response["result"]:
                tools = response["result"]["tools"]
                print(f"  Available tools: {len(tools)}")
                for tool in tools:
                    desc = tool.get('description', 'N/A')[:60]
                    print(f"    - {tool['name']}: {desc}...")
        else:
            print("\n[WARN] No tools response")
        
        # Terminate
        process.terminate()
        try:
            process.wait(timeout=3)
        except:
            process.kill()
        
        print("\n" + "=" * 60)
        print("ModelScope MCP Server test completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_modelscope_mcp()
    sys.exit(0 if success else 1)
