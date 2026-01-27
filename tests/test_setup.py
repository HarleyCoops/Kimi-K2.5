#!/usr/bin/env python3
"""
Tests for setup_api_key.py module
"""

import os
import sys
import pytest
from unittest.mock import patch, mock_open, MagicMock
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSetupApiKey:
    """Test setup_api_key functionality"""
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('setup_api_key.getpass.getpass')
    @patch('builtins.open', mock_open())
    @patch('setup_api_key.test_api_key')
    @patch('builtins.input')
    def test_setup_api_key_success(self, mock_input, mock_test, mock_getpass):
        mock_getpass.return_value = "valid_api_key_123"
        mock_input.return_value = "y"  # Confirm update
        mock_test.return_value = None  # Success
        
        import setup_api_key
        # Should not raise
        setup_api_key.setup_api_key()
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('setup_api_key.getpass.getpass')
    @patch('builtins.print')
    def test_setup_api_key_empty(self, mock_print, mock_getpass):
        mock_getpass.return_value = ""
        
        import setup_api_key
        setup_api_key.setup_api_key()
        
        # Should print error about empty key
        error_printed = any("No API key provided" in str(call) for call in mock_print.call_args_list)
        assert error_printed
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('setup_api_key.getpass.getpass')
    @patch('builtins.print')
    def test_setup_api_key_placeholder(self, mock_print, mock_getpass):
        mock_getpass.return_value = "your_api_key_here"
        
        import setup_api_key
        setup_api_key.setup_api_key()
        
        # Should print error about placeholder
        error_printed = any("Invalid API key" in str(call) for call in mock_print.call_args_list)
        assert error_printed


class TestTestApiKey:
    """Test test_api_key function"""
    
    @patch('openai.OpenAI')
    @patch('builtins.print')
    def test_test_api_key_success(self, mock_print, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        import setup_api_key
        setup_api_key.test_api_key("valid_key")
        
        # Should print success
        success_printed = any("validation completed successfully" in str(call) for call in mock_print.call_args_list)
        assert success_printed
    
    @patch('openai.OpenAI')
    @patch('builtins.print')
    def test_test_api_key_failure(self, mock_print, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock failed API call
        mock_client.chat.completions.create.side_effect = Exception("Invalid API key")
        
        import setup_api_key
        setup_api_key.test_api_key("invalid_key")
        
        # Should print error
        error_printed = any("validation failed" in str(call) for call in mock_print.call_args_list)
        assert error_printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
