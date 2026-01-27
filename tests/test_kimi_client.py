#!/usr/bin/env python3
"""
Tests for kimi_client.py module
"""

import os
import sys
import json
import base64
import pytest
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kimi_client import (
    KimiResponse,
    StreamChunk,
    KimiClient,
    create_client,
    instant,
    think,
)
from config import KimiMode, MODELS


class TestKimiResponse:
    """Test KimiResponse dataclass"""
    
    def test_basic_response(self):
        response = KimiResponse(content="Hello")
        assert response.content == "Hello"
        assert response.reasoning is None
        assert response.tool_calls is None
        assert response.finish_reason == "stop"
    
    def test_response_with_reasoning(self):
        response = KimiResponse(
            content="The answer is 4",
            reasoning="Let me think... 2+2=4",
            mode=KimiMode.THINKING
        )
        assert response.reasoning == "Let me think... 2+2=4"
        assert response.total_tokens == 0  # No usage data
    
    def test_has_tool_calls_property(self):
        response_with_tools = KimiResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[{"id": "1", "function": {"name": "test"}}]
        )
        assert response_with_tools.has_tool_calls is True
        
        response_without_tools = KimiResponse(content="Hello")
        assert response_without_tools.has_tool_calls is False
    
    def test_total_tokens_property(self):
        response = KimiResponse(
            content="Hello",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        assert response.total_tokens == 15
        
        response_no_usage = KimiResponse(content="Hello")
        assert response_no_usage.total_tokens == 0


class TestStreamChunk:
    """Test StreamChunk dataclass"""
    
    def test_default_values(self):
        chunk = StreamChunk()
        assert chunk.content == ""
        assert chunk.reasoning == ""
        assert chunk.tool_call_delta is None
        assert chunk.finish_reason is None
        assert chunk.is_reasoning is False


class TestKimiClientInitialization:
    """Test KimiClient initialization"""
    
    @patch('kimi_client.OpenAI')
    @patch('kimi_client.AsyncOpenAI')
    @patch.dict(os.environ, {"MOONSHOT_API_KEY": "test_key"})
    def test_default_initialization(self, mock_async_client, mock_sync_client):
        client = KimiClient()
        assert client.model_key == "k2.5"
        assert client.default_mode == KimiMode.THINKING
        mock_sync_client.assert_called_once()
    
    @patch('kimi_client.OpenAI')
    @patch('kimi_client.AsyncOpenAI')
    def test_custom_api_key(self, mock_async_client, mock_sync_client):
        # Set up the mock to return an instance with the api_key attribute
        mock_instance = MagicMock()
        mock_instance.api_key = "custom_key"
        mock_sync_client.return_value = mock_instance
        mock_async_client.return_value = MagicMock()
        
        client = KimiClient(api_key="custom_key")
        # Verify that OpenAI was called with the custom key
        mock_sync_client.assert_called_once()
        call_kwargs = mock_sync_client.call_args[1]
        assert call_kwargs['api_key'] == "custom_key"
    
    @patch('kimi_client.OpenAI')
    @patch('kimi_client.AsyncOpenAI')
    @patch.dict(os.environ, {"MOONSHOT_API_KEY": "test_key"})
    def test_custom_mode(self, mock_async_client, mock_sync_client):
        client = KimiClient(default_mode=KimiMode.AGENT)
        assert client.default_mode == KimiMode.AGENT


class TestKimiClientHelpers:
    """Test KimiClient helper methods"""
    
    @patch('kimi_client.OpenAI')
    @patch('kimi_client.AsyncOpenAI')
    @patch.dict(os.environ, {"MOONSHOT_API_KEY": "test_key"})
    def test_model_id_property(self, mock_async_client, mock_sync_client):
        client = KimiClient()
        assert client.model_id == MODELS["k2.5"].model_id
    
    @patch('kimi_client.OpenAI')
    @patch('kimi_client.AsyncOpenAI')
    @patch.dict(os.environ, {"MOONSHOT_API_KEY": "test_key"})
    def test_get_mode_params(self, mock_async_client, mock_sync_client):
        client = KimiClient()
        
        # Test instant mode params
        params = client._get_mode_params(KimiMode.INSTANT)
        assert "temperature" in params
        assert "max_tokens" in params
        assert "extra_body" in params
        
        # Test thinking mode params (no extra_body)
        params = client._get_mode_params(KimiMode.THINKING)
        assert "temperature" in params
        assert "extra_body" not in params


class TestBuildMessages:
    """Test _build_messages method"""
    
    @patch('kimi_client.OpenAI')
    @patch('kimi_client.AsyncOpenAI')
    @patch.dict(os.environ, {"MOONSHOT_API_KEY": "test_key"})
    def test_basic_message(self, mock_async_client, mock_sync_client):
        client = KimiClient()
        messages = client._build_messages("Hello", mode=KimiMode.INSTANT)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
    
    @patch('kimi_client.OpenAI')
    @patch('kimi_client.AsyncOpenAI')
    @patch.dict(os.environ, {"MOONSHOT_API_KEY": "test_key"})
    def test_with_custom_system_prompt(self, mock_async_client, mock_sync_client):
        client = KimiClient()
        messages = client._build_messages(
            "Hello",
            system_prompt="Custom prompt",
            mode=KimiMode.INSTANT
        )
        
        assert messages[0]["content"] == "Custom prompt"
    
    @patch('kimi_client.OpenAI')
    @patch('kimi_client.AsyncOpenAI')
    @patch.dict(os.environ, {"MOONSHOT_API_KEY": "test_key"})
    def test_with_history(self, mock_async_client, mock_sync_client):
        client = KimiClient()
        history = [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"}
        ]
        messages = client._build_messages("Hello", mode=KimiMode.INSTANT, history=history)
        
        assert len(messages) == 4
        assert messages[1] == history[0]
        assert messages[2] == history[1]


class TestImageEncoding:
    """Test image encoding functionality"""
    
    @patch('kimi_client.OpenAI')
    @patch('kimi_client.AsyncOpenAI')
    @patch.dict(os.environ, {"MOONSHOT_API_KEY": "test_key"})
    def test_encode_image_not_found(self, mock_async_client, mock_sync_client):
        client = KimiClient()
        with pytest.raises(FileNotFoundError):
            client._encode_image("/nonexistent/image.png")
    
    @patch('kimi_client.OpenAI')
    @patch('kimi_client.AsyncOpenAI')
    @patch.dict(os.environ, {"MOONSHOT_API_KEY": "test_key"})
    @patch('builtins.open', mock_open(read_data=b'fake_image_data'))
    @patch('pathlib.Path.exists')
    def test_encode_image_success(self, mock_exists, mock_async_client, mock_sync_client):
        mock_exists.return_value = True
        client = KimiClient()
        
        # Create a mock path with suffix
        from pathlib import Path
        with patch.object(Path, 'suffix', '.png'):
            result = client._encode_image("test.png")
            assert result.startswith("data:image/png;base64,")


class TestParseResponse:
    """Test _parse_response method"""
    
    @patch('kimi_client.OpenAI')
    @patch('kimi_client.AsyncOpenAI')
    @patch.dict(os.environ, {"MOONSHOT_API_KEY": "test_key"})
    def test_parse_basic_response(self, mock_async_client, mock_sync_client):
        client = KimiClient()
        
        # Mock response object
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = None
        mock_response.model = "moonshotai/Kimi-K2.5"
        
        result = client._parse_response(mock_response, KimiMode.THINKING)
        
        assert result.content == "Hello"
        assert result.finish_reason == "stop"
        assert result.mode == KimiMode.THINKING


class TestCreateClient:
    """Test create_client convenience function"""
    
    @patch('kimi_client.OpenAI')
    @patch('kimi_client.AsyncOpenAI')
    @patch.dict(os.environ, {"MOONSHOT_API_KEY": "test_key"})
    def test_create_client_with_mode(self, mock_async_client, mock_sync_client):
        client = create_client(mode=KimiMode.AGENT)
        assert client.default_mode == KimiMode.AGENT


@pytest.mark.skip(reason="Requires actual API key")
class TestIntegration:
    """Integration tests - skipped by default"""
    
    def test_actual_api_call(self):
        """This would test against the real API"""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
