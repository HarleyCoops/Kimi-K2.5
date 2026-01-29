#!/usr/bin/env python3
"""
Debug ModelScope MCP Server
"""

import os
import subprocess
import sys
import threading
import time

def read_output(pipe, prefix):
    """Read output from pipe"""
    for line in iter(pipe.readline, b''):
        print(f"[{prefix}] {line.decode('utf-8', errors='replace').rstrip()}")
    pipe.close()

def test_modelscope():
    """Test ModelScope MCP server"""

    api_token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not api_token:
        raise SystemExit(
            "MODELSCOPE_API_TOKEN is not set. "
            "Set it in your environment before running this test."
        )
    
    print("=" * 60)
    print("Debugging ModelScope MCP Server")
    print("=" * 60)
    
    env = {"MODELSCOPE_API_TOKEN": api_token}
    env["PATH"] = subprocess.os.environ.get("PATH", "")
    
    try:
        process = subprocess.Popen(
            ["uvx", "modelscope-mcp-server", "--transport", "sse", "--port", "8000"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Start threads to read output
        threading.Thread(target=read_output, args=(process.stdout, "OUT"), daemon=True).start()
        threading.Thread(target=read_output, args=(process.stderr, "ERR"), daemon=True).start()
        
        print("\nServer starting... waiting 5 seconds for output\n")
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("\n[OK] Server is running")
        else:
            print(f"\n[ERROR] Server exited with code: {process.returncode}")
        
        # Terminate
        process.terminate()
        try:
            process.wait(timeout=2)
        except:
            process.kill()
            
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_modelscope()
