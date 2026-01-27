#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_api_key():
    """Fixture to provide a mock API key"""
    return "test_api_key_12345"


@pytest.fixture
def mock_env_with_key(mock_api_key):
    """Fixture to set up environment with mock API key"""
    with patch.dict(os.environ, {"MOONSHOT_API_KEY": mock_api_key}, clear=False):
        yield mock_api_key


@pytest.fixture
def mock_openai_client():
    """Fixture to provide a mocked OpenAI client"""
    with patch('kimi_client.OpenAI') as mock_sync, \
         patch('kimi_client.AsyncOpenAI') as mock_async:
        
        mock_sync_instance = MagicMock()
        mock_async_instance = MagicMock()
        mock_sync.return_value = mock_sync_instance
        mock_async.return_value = mock_async_instance
        
        yield {
            'sync': mock_sync_instance,
            'async': mock_async_instance,
            'sync_class': mock_sync,
            'async_class': mock_async
        }


@pytest.fixture
def sample_tool_definitions():
    """Fixture to provide sample tool definitions"""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "required": ["city"],
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform calculation",
                "parameters": {
                    "type": "object",
                    "required": ["expression"],
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression"
                        }
                    }
                }
            }
        }
    ]


@pytest.fixture
def sample_chat_response():
    """Fixture to provide a sample chat completion response"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.choices[0].message.reasoning_content = None
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model = "moonshotai/Kimi-K2.5"
    return mock_response


@pytest.fixture
def sample_tool_call_response():
    """Fixture to provide a sample tool call response"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None
    mock_response.choices[0].message.reasoning_content = None
    
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.type = "function"
    mock_tool_call.function.name = "get_weather"
    mock_tool_call.function.arguments = '{"city": "Beijing"}'
    
    mock_response.choices[0].message.tool_calls = [mock_tool_call]
    mock_response.choices[0].finish_reason = "tool_calls"
    mock_response.usage = None
    mock_response.model = "moonshotai/Kimi-K2.5"
    return mock_response
