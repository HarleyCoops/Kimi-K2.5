#!/usr/bin/env python3
"""
Tests for config.py module
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Ensure we can import from parent directory
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    KimiMode,
    ModeConfig,
    MODE_CONFIGS,
    MODELS,
    DEFAULT_MODEL,
    DEFAULT_MODE,
    ModelInfo,
    APIConfig,
    SwarmConfig,
    MultimodalConfig,
    ToolConfig,
    SYSTEM_PROMPTS,
    get_mode_config,
    get_model_info,
    validate_api_key,
)


class TestKimiMode:
    """Test KimiMode enum"""
    
    def test_mode_values(self):
        assert KimiMode.INSTANT.value == "instant"
        assert KimiMode.THINKING.value == "thinking"
        assert KimiMode.AGENT.value == "agent"
        assert KimiMode.SWARM.value == "swarm"


class TestModeConfigs:
    """Test mode configurations"""
    
    def test_all_modes_have_config(self):
        for mode in KimiMode:
            assert mode in MODE_CONFIGS
    
    def test_instant_mode_config(self):
        config = MODE_CONFIGS[KimiMode.INSTANT]
        assert config.temperature == 0.6
        assert config.max_tokens == 4096
        assert config.extra_body == {"thinking": {"type": "disabled"}}
    
    def test_thinking_mode_config(self):
        config = MODE_CONFIGS[KimiMode.THINKING]
        assert config.temperature == 1.0
        assert config.max_tokens == 8192
        assert config.extra_body is None
    
    def test_agent_mode_config(self):
        config = MODE_CONFIGS[KimiMode.AGENT]
        assert config.temperature == 0.6
        assert config.max_tokens == 4096
    
    def test_swarm_mode_config(self):
        config = MODE_CONFIGS[KimiMode.SWARM]
        assert config.temperature == 1.0
        assert config.max_tokens == 16384


class TestModels:
    """Test model definitions"""
    
    def test_k2_5_model_exists(self):
        assert "k2.5" in MODELS
        model = MODELS["k2.5"]
        assert model.model_id == "kimi-k2.5"
        assert model.supports_vision is True
        assert model.supports_video is True
    
    def test_legacy_models_exist(self):
        legacy_models = ["k2-instruct", "k2-thinking", "k2-base"]
        for model_key in legacy_models:
            assert model_key in MODELS
    
    def test_default_model_is_k2_5(self):
        assert DEFAULT_MODEL == "k2.5"


class TestGetModeConfig:
    """Test get_mode_config function"""
    
    def test_returns_correct_config(self):
        for mode in KimiMode:
            config = get_mode_config(mode)
            assert isinstance(config, ModeConfig)
            assert config == MODE_CONFIGS[mode]


class TestGetModelInfo:
    """Test get_model_info function"""
    
    def test_returns_correct_model_info(self):
        info = get_model_info("k2.5")
        assert isinstance(info, ModelInfo)
        assert info.model_id == "kimi-k2.5"
    
    def test_invalid_model_raises_error(self):
        with pytest.raises(ValueError) as exc_info:
            get_model_info("invalid-model")
        assert "Unknown model" in str(exc_info.value)
    
    def test_default_model(self):
        info = get_model_info()
        assert info.model_id == MODELS[DEFAULT_MODEL].model_id


class TestValidateApiKey:
    """Test validate_api_key function"""
    
    def test_returns_false_when_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            assert validate_api_key() is False
    
    def test_returns_true_when_set(self):
        with patch.dict(os.environ, {"MOONSHOT_API_KEY": "test_key"}):
            assert validate_api_key() is True
    
    def test_returns_false_for_empty_string(self):
        with patch.dict(os.environ, {"MOONSHOT_API_KEY": ""}):
            assert validate_api_key() is False


class TestSwarmConfig:
    """Test SwarmConfig dataclass"""
    
    def test_default_values(self):
        config = SwarmConfig()
        assert config.max_agents == 100
        assert config.max_parallel_tool_calls == 1500
        assert config.main_agent_max_steps == 15
        assert config.coordination_timeout == 300
    
    def test_custom_values(self):
        config = SwarmConfig(max_agents=50, coordination_timeout=600)
        assert config.max_agents == 50
        assert config.coordination_timeout == 600
        # Other values should remain default
        assert config.max_parallel_tool_calls == 1500


class TestMultimodalConfig:
    """Test MultimodalConfig dataclass"""
    
    def test_default_values(self):
        config = MultimodalConfig()
        assert config.max_image_size_mb == 20.0
        assert config.max_video_size_mb == 100.0
        assert "png" in config.supported_image_formats
        assert "mp4" in config.supported_video_formats
        assert config.image_detail == "auto"


class TestToolConfig:
    """Test ToolConfig dataclass"""
    
    def test_default_values(self):
        config = ToolConfig()
        assert config.max_tool_steps == 300
        assert config.parallel_execution is True
        assert config.max_concurrent_calls == 100
        assert config.tool_timeout == 60


class TestSystemPrompts:
    """Test system prompts for each mode"""
    
    def test_all_modes_have_prompt(self):
        for mode in KimiMode:
            assert mode in SYSTEM_PROMPTS
            assert isinstance(SYSTEM_PROMPTS[mode], str)
            assert len(SYSTEM_PROMPTS[mode]) > 0
    
    def test_prompts_contain_kimi_reference(self):
        for mode, prompt in SYSTEM_PROMPTS.items():
            assert "Kimi" in prompt or "Moonshot" in prompt


class TestAPIConfig:
    """Test APIConfig dataclass"""
    
    def test_from_env_with_valid_key(self):
        with patch.dict(os.environ, {"MOONSHOT_API_KEY": "test_key_123"}):
            config = APIConfig.from_env()
            assert config.api_key == "test_key_123"
            assert config.base_url == "https://api.moonshot.ai/v1"
            assert config.timeout == 120
            assert config.max_retries == 3
    
    def test_from_env_with_custom_base_url(self):
        with patch.dict(os.environ, {
            "MOONSHOT_API_KEY": "test_key",
            "KIMI_API_BASE_URL": "https://custom.api.com/v1"
        }):
            config = APIConfig.from_env()
            assert config.base_url == "https://custom.api.com/v1"
    
    def test_from_env_raises_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                APIConfig.from_env()
            assert "MOONSHOT_API_KEY" in str(exc_info.value)
    
    def test_from_env_with_custom_timeout_and_retries(self):
        with patch.dict(os.environ, {
            "MOONSHOT_API_KEY": "test_key",
            "KIMI_API_TIMEOUT": "60",
            "KIMI_API_MAX_RETRIES": "5"
        }):
            config = APIConfig.from_env()
            assert config.timeout == 60
            assert config.max_retries == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
