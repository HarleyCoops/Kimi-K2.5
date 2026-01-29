#!/usr/bin/env python3
"""
Test ModelScope MCP Server via HTTP transport
"""

import json
import requests
import subprocess
import sys
import time

def test_modelscope_http():
    """Test ModelScope MCP server via HTTP"""
    
    api_token = "sk-kimi-uXQjtFdVGqyz67t3NYh3wFdU0VerKbiUqGt2Ffef1rQ4WYxGdga8T2NnM01Cf7OI"
    port = 8000
    
    print("=" * 60)
    print("Testing ModelScope MCP Server (HTTP mode)")
    print("=" * 60)
    
    # Start the MCP server in HTTP mode
    env = {"MODELSCOPE_API_TOKEN": api_token, "PATH": subprocess.os.environ.get("PATH", "")}
    
    try:
        process = subprocess.Popen(
            ["uvx", "modelscope-mcp-server", "--transport", "http", "--port", str(port)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Wait for server to start
        print(f"\nStarting server on port {port}...")
        time.sleep(3)
        
        base_url = f"http://localhost:{port}"
        
        # Test health endpoint
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            print(f"\n[OK] Health check: {response.status_code}")
        except Exception as e:
            print(f"\n[WARN] Health check failed: {e}")
        
        # Test SSE endpoint for MCP
        try:
            response = requests.get(f"{base_url}/sse", timeout=5, stream=True)
            print(f"[OK] SSE endpoint: {response.status_code}")
        except Exception as e:
            print(f"[WARN] SSE endpoint: {e}")
        
        # Test tools via HTTP POST (if supported)
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            }
            response = requests.post(
                f"{base_url}/message",
                headers=headers,
                json=data,
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                print(f"\n[OK] Tools list retrieved")
                if "result" in result and "tools" in result["result"]:
                    tools = result["result"]["tools"]
                    print(f"  Available tools: {len(tools)}")
                    for tool in tools:
                        desc = tool.get('description', 'N/A')[:60]
                        print(f"    - {tool['name']}: {desc}")
            else:
                print(f"\n[WARN] Tools request: {response.status_code}")
                print(f"  Response: {response.text[:200]}")
        except Exception as e:
            print(f"\n[WARN] Tools request failed: {e}")
        
        # Cleanup
        process.terminate()
        try:
            process.wait(timeout=3)
        except:
            process.kill()
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_modelscope_http()
    sys.exit(0 if success else 1)
