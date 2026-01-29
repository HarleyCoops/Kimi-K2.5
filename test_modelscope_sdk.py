#!/usr/bin/env python3
"""
Test ModelScope Python SDK
"""

import sys

def test_modelscope_sdk():
    """Test ModelScope SDK"""
    
    print("=" * 60)
    print("Testing ModelScope Python SDK")
    print("=" * 60)
    
    # Check if modelscope is installed
    try:
        import modelscope
        print(f"\n[OK] ModelScope SDK installed: {modelscope.__version__}")
    except ImportError:
        print("\n[INFO] ModelScope SDK not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope", "-q"])
        import modelscope
        print(f"[OK] ModelScope SDK installed: {modelscope.__version__}")
    
    # Try to use the SDK
    print("\n[Test] Searching for models via SDK...")
    try:
        from modelscope.hub.api import HubApi
        
        api = HubApi()
        
        # Search for kimi models
        models = api.list_models(search_keyword="kimi", page_size=5)
        print(f"[OK] Found {len(models)} models")
        
        for m in models[:3]:
            print(f"  - {m.get('ModelId', 'N/A')}: {m.get('Name', 'N/A')[:40]}")
            
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    # Try with API token
    print("\n[Test] Testing with API token...")
    try:
        from modelscope.hub.api import HubApi
        
        api_token = "sk-kimi-uXQjtFdVGqyz67t3NYh3wFdU0VerKbiUqGt2Ffef1rQ4WYxGdga8T2NnM01Cf7OI"
        api = HubApi()
        api.login(api_token)
        
        # Get user info
        user_info = api.get_user_info()
        print(f"[OK] Logged in successfully")
        print(f"  User: {user_info}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("SDK Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_modelscope_sdk()
