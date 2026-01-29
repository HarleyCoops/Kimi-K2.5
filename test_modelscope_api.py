#!/usr/bin/env python3
"""
Test ModelScope API directly
"""

import requests
import json

def test_modelscope_api():
    """Test ModelScope API with the token"""
    
    api_token = "sk-kimi-uXQjtFdVGqyz67t3NYh3wFdU0VerKbiUqGt2Ffef1rQ4WYxGdga8T2NnM01Cf7OI"
    
    print("=" * 60)
    print("Testing ModelScope API Directly")
    print("=" * 60)
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Test 1: Get user info
    print("\n[Test 1] Getting user info...")
    try:
        response = requests.get(
            "https://www.modelscope.cn/api/v1/my/user",
            headers=headers,
            timeout=10
        )
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  [OK] User info retrieved")
            print(f"  Data: {json.dumps(data, indent=2)[:300]}...")
        else:
            print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test 2: Search for models
    print("\n[Test 2] Searching for models...")
    try:
        response = requests.get(
            "https://www.modelscope.cn/api/v1/dolphin/models",
            headers=headers,
            params={"Search": "kimi", "PageSize": 5},
            timeout=10
        )
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  [OK] Models search successful")
            if "Data" in data:
                models = data["Data"]
                print(f"  Found {len(models)} models")
                for m in models[:3]:
                    print(f"    - {m.get('ModelId', 'N/A')}: {m.get('Name', 'N/A')[:50]}")
        else:
            print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test 3: Try inference API (if available)
    print("\n[Test 3] Testing inference endpoint...")
    try:
        # This is a common ModelScope inference endpoint format
        response = requests.post(
            "https://www.modelscope.cn/api/v1/studio/damo/nlp_kimi/gradio/api/predict",
            headers=headers,
            json={"fn_index": 0, "data": ["Hello, how are you?"]},
            timeout=10
        )
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  [OK] Inference endpoint responded")
            print(f"  Response: {response.text[:200]}")
        else:
            print(f"  [INFO] Endpoint returned {response.status_code} (may be expected)")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    print("\n" + "=" * 60)
    print("API Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_modelscope_api()
