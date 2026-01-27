#!/usr/bin/env python3
"""
Kimi K2.5 Configuration Module

This module provides centralized configuration for all Kimi K2.5 capabilities:
- Model variants and mode configurations
- API endpoints and authentication
- Default parameters for different use cases
- Agent swarm settings
- Multimodal processing options

References:
- Official API: https://platform.moonshot.ai
- Model Card: https://huggingface.co/moonshotai/Kimi-K2.5
- Tech Blog: https://www.kimi.com/blog/kimi-k2-5.html
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class KimiMode(Enum):
    """
    Kimi K2.5 Operating Modes

    K2.5 supports four distinct operating modes, each optimized for different tasks:

    INSTANT: Fast responses without reasoning traces
        - Best for: Quick queries, simple tasks, low-latency requirements
        - Temperature: 0.6
        - Thinking: Disabled

    THINKING: Deep reasoning with explicit thought traces
        - Best for: Complex problems, math, coding, research
        - Temperature: 1.0
        - Output: Includes reasoning_content field

    AGENT: Single agent with tool calling
        - Best for: Autonomous tasks with tool use
        - Temperature: 0.6
        - Supports: 200-300 consecutive tool invocations

    SWARM: Multi-agent orchestration
        - Best for: Complex parallel workflows
        - Agents: Up to 100 sub-agents
        - Tool calls: Up to 1,500 parallel executions
        - Speedup: Up to 4.5x vs single agent
    """
    INSTANT = "instant"
    THINKING = "thinking"
    AGENT = "agent"
    SWARM = "swarm"


@dataclass
class ModeConfig:
    """Configuration for a specific Kimi K2.5 mode"""
    temperature: float
    max_tokens: int
    top_p: float = 0.95
    extra_body: Optional[Dict[str, Any]] = None
    description: str = ""


# Mode-specific configurations based on official recommendations
MODE_CONFIGS: Dict[KimiMode, ModeConfig] = {
    KimiMode.INSTANT: ModeConfig(
        temperature=0.6,
        max_tokens=4096,
        extra_body={"thinking": {"type": "disabled"}},
        description="Fast responses without reasoning traces. Best for quick queries."
    ),
    KimiMode.THINKING: ModeConfig(
        temperature=1.0,
        max_tokens=8192,
        extra_body=None,  # Thinking is enabled by default
        description="Deep reasoning with thought traces. Best for complex problems."
    ),
    KimiMode.AGENT: ModeConfig(
        temperature=1.0,  # API requires temperature=1.0 for kimi-k2.5
        max_tokens=4096,
        extra_body=None,
        description="Single agent with extended tool calling. Supports 200-300 tool steps."
    ),
    KimiMode.SWARM: ModeConfig(
        temperature=1.0,
        max_tokens=16384,
        extra_body=None,
        description="Multi-agent orchestration. Up to 100 agents, 1,500 parallel tool calls."
    ),
}


@dataclass
class ModelInfo:
    """Information about a Kimi model variant"""
    model_id: str
    display_name: str
    description: str
    context_length: int
    max_completion_tokens: int
    supports_vision: bool = False
    supports_video: bool = False
    supports_tools: bool = True
    supports_streaming: bool = True


# Available Kimi K2.5 model variants
MODELS = {
    # Primary K2.5 model (multimodal, all modes)
    "k2.5": ModelInfo(
        model_id="kimi-k2.5",
        display_name="Kimi K2.5",
        description="Native multimodal model with visual coding and agent swarm capabilities",
        context_length=262144,  # 256K tokens
        max_completion_tokens=96000,  # 96K for reasoning tasks
        supports_vision=True,
        supports_video=True,  # Only on official API
    ),

    # K2 Instruct model (currently active on Moonshot API)
    "k2-instruct": ModelInfo(
        model_id="moonshotai/Kimi-K2-Instruct",
        display_name="Kimi K2 Instruct",
        description="Post-trained model for general-purpose chat and agentic experiences",
        context_length=131072,  # 128K tokens
        max_completion_tokens=8192,
        supports_vision=False,
        supports_video=False,
    ),
    "k2-thinking": ModelInfo(
        model_id="moonshotai/Kimi-K2-Thinking",
        display_name="Kimi K2 Thinking",
        description="Advanced reasoning model with extended thinking capabilities",
        context_length=131072,
        max_completion_tokens=32768,
        supports_vision=False,
        supports_video=False,
    ),
    "k2-base": ModelInfo(
        model_id="moonshotai/Kimi-K2-Base",
        display_name="Kimi K2 Base",
        description="Foundation model for fine-tuning and custom solutions",
        context_length=131072,
        max_completion_tokens=8192,
        supports_vision=False,
        supports_video=False,
        supports_tools=False,
    ),
}

# Default model for K2.5 operations
DEFAULT_MODEL = "k2.5"
DEFAULT_MODE = KimiMode.THINKING


@dataclass
class APIConfig:
    """API configuration settings"""
    base_url: str
    api_key: str
    timeout: int = 120
    max_retries: int = 3

    @classmethod
    def from_env(cls, base_url: Optional[str] = None) -> "APIConfig":
        """Create APIConfig from environment variables"""
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError(
                "MOONSHOT_API_KEY environment variable not set. "
                "Get your API key from https://platform.moonshot.ai"
            )
        return cls(
            base_url=base_url or os.getenv("KIMI_API_BASE_URL", "https://api.moonshot.ai/v1"),
            api_key=api_key,
            timeout=int(os.getenv("KIMI_API_TIMEOUT", "120")),
            max_retries=int(os.getenv("KIMI_API_MAX_RETRIES", "3")),
        )


# Alternative API providers (OpenRouter, AIML API, etc.)
API_PROVIDERS = {
    "moonshot": {
        "base_url": "https://api.moonshot.ai/v1",
        "env_key": "MOONSHOT_API_KEY",
        "description": "Official Moonshot AI API (recommended)",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "description": "OpenRouter API (multi-provider routing)",
    },
    "aimlapi": {
        "base_url": "https://api.aimlapi.com/v1",
        "env_key": "AIMLAPI_KEY",
        "description": "AIML API provider",
    },
}


@dataclass
class SwarmConfig:
    """Configuration for Agent Swarm operations"""
    max_agents: int = 100
    max_parallel_tool_calls: int = 1500
    main_agent_max_steps: int = 15
    sub_agent_max_steps: int = 100
    coordination_timeout: int = 300  # 5 minutes
    result_aggregation_strategy: str = "synthesis"  # "synthesis", "merge", "vote"

    # Sub-agent creation settings
    dynamic_agent_creation: bool = True
    agent_specializations: List[str] = field(default_factory=lambda: [
        "researcher",
        "coder",
        "analyst",
        "verifier",
        "writer",
    ])


@dataclass
class MultimodalConfig:
    """Configuration for multimodal (vision/video) operations"""
    max_image_size_mb: float = 20.0
    max_video_size_mb: float = 100.0
    supported_image_formats: List[str] = field(default_factory=lambda: [
        "png", "jpg", "jpeg", "gif", "webp", "bmp"
    ])
    supported_video_formats: List[str] = field(default_factory=lambda: [
        "mp4", "webm", "avi", "mov"
    ])
    image_detail: str = "auto"  # "auto", "low", "high"

    # Vision encoder info (MoonViT)
    vision_encoder: str = "MoonViT"
    vision_encoder_params: int = 400_000_000  # 400M parameters
    vision_hidden_dim: int = 7168


@dataclass
class ToolConfig:
    """Configuration for tool calling operations"""
    max_tool_steps: int = 300
    parallel_execution: bool = True
    max_concurrent_calls: int = 100
    tool_timeout: int = 60
    retry_on_failure: bool = True
    max_tool_retries: int = 2

    # Interleaved thinking settings
    log_reasoning: bool = True
    reasoning_detail_level: str = "full"  # "full", "summary", "none"


# Default configurations
DEFAULT_API_CONFIG = APIConfig.from_env() if os.getenv("MOONSHOT_API_KEY") else None
DEFAULT_SWARM_CONFIG = SwarmConfig()
DEFAULT_MULTIMODAL_CONFIG = MultimodalConfig()
DEFAULT_TOOL_CONFIG = ToolConfig()


# System prompts for different modes
SYSTEM_PROMPTS = {
    KimiMode.INSTANT: (
        "You are Kimi, an AI assistant created by Moonshot AI. "
        "Provide concise, accurate responses efficiently."
    ),
    KimiMode.THINKING: (
        "You are Kimi, an AI assistant created by Moonshot AI. "
        "Think step by step and show your reasoning process clearly. "
        "For complex problems, break them down into smaller parts."
    ),
    KimiMode.AGENT: (
        "You are Kimi, an autonomous AI agent created by Moonshot AI. "
        "You have access to tools and can execute multi-step workflows. "
        "Plan your actions carefully and use tools when needed to complete tasks."
    ),
    KimiMode.SWARM: (
        "You are the orchestrator of a Kimi Agent Swarm. "
        "Decompose complex tasks into parallelizable subtasks. "
        "Create and coordinate specialized sub-agents efficiently. "
        "Synthesize results from multiple agents into coherent outputs."
    ),
}


# Model architecture details (for reference)
MODEL_ARCHITECTURE = {
    "type": "Mixture-of-Experts (MoE)",
    "total_parameters": 1_000_000_000_000,  # 1T
    "activated_parameters": 32_000_000_000,  # 32B
    "layers": 61,
    "dense_layers": 1,
    "attention_hidden_dim": 7168,
    "moe_hidden_dim_per_expert": 2048,
    "attention_heads": 64,
    "num_experts": 384,
    "selected_experts_per_token": 8,
    "shared_experts": 1,
    "vocabulary_size": 160_000,
    "attention_mechanism": "MLA (Multi-Head Latent Attention)",
    "activation_function": "SwiGLU",
    "optimizer": "MuonClip",
    "training_tokens": 15_500_000_000_000,  # 15.5T tokens
}


def get_mode_config(mode: KimiMode) -> ModeConfig:
    """Get configuration for a specific mode"""
    return MODE_CONFIGS[mode]


def get_model_info(model_key: str = DEFAULT_MODEL) -> ModelInfo:
    """Get information about a model variant"""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
    return MODELS[model_key]


def validate_api_key() -> bool:
    """Check if a valid API key is configured"""
    return bool(os.getenv("MOONSHOT_API_KEY"))


def print_config_summary():
    """Print a summary of current configuration"""
    print("=" * 60)
    print("KIMI K2.5 CONFIGURATION SUMMARY")
    print("=" * 60)

    print(f"\nDefault Model: {MODELS[DEFAULT_MODEL].display_name}")
    print(f"  ID: {MODELS[DEFAULT_MODEL].model_id}")
    print(f"  Context: {MODELS[DEFAULT_MODEL].context_length:,} tokens")
    print(f"  Vision: {'Yes' if MODELS[DEFAULT_MODEL].supports_vision else 'No'}")

    print(f"\nDefault Mode: {DEFAULT_MODE.value}")
    mode_cfg = MODE_CONFIGS[DEFAULT_MODE]
    print(f"  Temperature: {mode_cfg.temperature}")
    print(f"  Max Tokens: {mode_cfg.max_tokens:,}")

    print(f"\nAPI Key: {'Configured' if validate_api_key() else 'NOT SET'}")

    print("\nAvailable Modes:")
    for mode, cfg in MODE_CONFIGS.items():
        print(f"  - {mode.value}: {cfg.description[:50]}...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_config_summary()
