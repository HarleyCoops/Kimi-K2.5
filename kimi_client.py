#!/usr/bin/env python3
"""
Kimi K2.5 Unified Client

A production-ready client for Kimi K2.5 supporting all operating modes:
- Instant: Fast responses without reasoning traces
- Thinking: Deep reasoning with explicit thought chains
- Agent: Single agent with extended tool calling (200-300 steps)
- Swarm: Multi-agent orchestration (100 agents, 1,500 parallel tool calls)

Features:
- Mode switching with automatic configuration
- Multimodal inputs (images, video)
- Streaming with reasoning token handling
- Comprehensive error handling and retries
- Token counting and cost estimation

Example Usage:
    from kimi_client import KimiClient, KimiMode

    client = KimiClient()

    # Quick response (Instant mode)
    response = client.chat("What is 2+2?", mode=KimiMode.INSTANT)

    # Deep reasoning (Thinking mode)
    response = client.chat("Prove the Pythagorean theorem", mode=KimiMode.THINKING)
    print(response.reasoning)  # Access reasoning trace

    # With image input
    response = client.chat_with_image("Describe this UI", "screenshot.png")

    # Streaming
    for chunk in client.stream("Write a story"):
        print(chunk.content, end="")
"""

import json
import base64
import asyncio
from pathlib import Path
from typing import (
    Dict, List, Optional, Any, Union, Generator, AsyncGenerator,
    Callable, TypeVar, Literal
)
from dataclasses import dataclass, field
import logging

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    KimiMode, ModeConfig, MODE_CONFIGS, MODELS, DEFAULT_MODEL, DEFAULT_MODE,
    APIConfig, SYSTEM_PROMPTS, MultimodalConfig, ToolConfig,
    DEFAULT_MULTIMODAL_CONFIG, DEFAULT_TOOL_CONFIG
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KimiResponse:
    """
    Structured response from Kimi K2.5

    Attributes:
        content: The main response text
        reasoning: Reasoning trace (Thinking mode only)
        tool_calls: List of tool calls if any
        finish_reason: Why the response ended
        usage: Token usage statistics
        model: Model used for this response
        mode: Operating mode used
    """
    content: str
    reasoning: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"
    usage: Optional[Dict[str, int]] = None
    model: str = ""
    mode: KimiMode = KimiMode.THINKING

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls"""
        return self.finish_reason == "tool_calls" and bool(self.tool_calls)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used"""
        return self.usage.get("total_tokens", 0) if self.usage else 0


@dataclass
class StreamChunk:
    """A chunk from streaming response"""
    content: str = ""
    reasoning: str = ""
    tool_call_delta: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    is_reasoning: bool = False


class KimiClient:
    """
    Unified client for Kimi K2.5 API

    Supports all K2.5 operating modes with automatic configuration,
    multimodal inputs, streaming, and comprehensive tool calling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        default_mode: KimiMode = DEFAULT_MODE,
        multimodal_config: Optional[MultimodalConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ):
        """
        Initialize the Kimi client.

        Args:
            api_key: API key (defaults to MOONSHOT_API_KEY env var)
            base_url: API base URL (defaults to official Moonshot API)
            model: Model variant to use (default: k2.5)
            default_mode: Default operating mode
            multimodal_config: Settings for image/video processing
            tool_config: Settings for tool calling
        """
        # Load API configuration
        config = APIConfig.from_env(base_url)
        if api_key:
            config.api_key = api_key

        # Initialize clients
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
        self.async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

        # Store configuration
        self.model_key = model
        self.model_info = MODELS[model]
        self.default_mode = default_mode
        self.multimodal_config = multimodal_config or DEFAULT_MULTIMODAL_CONFIG
        self.tool_config = tool_config or DEFAULT_TOOL_CONFIG

        logger.info(f"Initialized KimiClient with model: {self.model_info.display_name}")

    @property
    def model_id(self) -> str:
        """Get the current model ID"""
        return self.model_info.model_id

    def _get_mode_params(self, mode: KimiMode) -> Dict[str, Any]:
        """Get API parameters for a specific mode"""
        config = MODE_CONFIGS[mode]
        params = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        }
        if config.extra_body:
            params["extra_body"] = config.extra_body
        return params

    def _build_messages(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        mode: KimiMode = None,
        history: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Build the messages array for API request"""
        mode = mode or self.default_mode
        messages = []

        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": SYSTEM_PROMPTS[mode]})

        # Add history
        if history:
            messages.extend(history)

        # Build user message content
        if images:
            content = [{"type": "text", "text": user_message}]
            for image_path in images:
                image_data = self._encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_message})

        return messages

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 data URL"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Determine MIME type
        suffix = path.suffix.lower().lstrip(".")
        mime_map = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
        }
        mime_type = mime_map.get(suffix, "image/png")

        # Encode
        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        return f"data:{mime_type};base64,{image_data}"

    def _encode_video(self, video_path: str) -> str:
        """Encode video to base64 data URL"""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        suffix = path.suffix.lower().lstrip(".")
        mime_map = {
            "mp4": "video/mp4",
            "webm": "video/webm",
            "avi": "video/avi",
            "mov": "video/quicktime",
        }
        mime_type = mime_map.get(suffix, "video/mp4")

        with open(path, "rb") as f:
            video_data = base64.b64encode(f.read()).decode()

        return f"data:{mime_type};base64,{video_data}"

    def _parse_response(
        self,
        response: ChatCompletion,
        mode: KimiMode
    ) -> KimiResponse:
        """Parse API response into KimiResponse"""
        choice = response.choices[0]
        message = choice.message

        # Extract content and reasoning
        content = message.content or ""
        reasoning = None
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            reasoning = message.reasoning_content

        # Extract tool calls
        tool_calls = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in message.tool_calls
            ]

        # Extract usage
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return KimiResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
            model=response.model,
            mode=mode,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def chat(
        self,
        message: str,
        mode: Optional[KimiMode] = None,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        **kwargs
    ) -> KimiResponse:
        """
        Send a chat message and get a response.

        Args:
            message: The user message
            mode: Operating mode (default: self.default_mode)
            system_prompt: Custom system prompt (optional)
            history: Conversation history
            tools: Tool definitions for function calling
            tool_choice: How to choose tools ("auto", "none", "required")
            **kwargs: Additional API parameters

        Returns:
            KimiResponse with content, reasoning (if applicable), and tool calls
        """
        mode = mode or self.default_mode
        messages = self._build_messages(message, system_prompt, mode, history)

        # Build API parameters
        params = {
            "model": self.model_id,
            "messages": messages,
            **self._get_mode_params(mode),
            **kwargs
        }

        # Add tools if provided
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        # Make request
        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, mode)

    def chat_with_image(
        self,
        message: str,
        image_paths: Union[str, List[str]],
        mode: Optional[KimiMode] = None,
        **kwargs
    ) -> KimiResponse:
        """
        Chat with image input (multimodal).

        Args:
            message: The user message
            image_paths: Path(s) to image file(s)
            mode: Operating mode
            **kwargs: Additional parameters

        Returns:
            KimiResponse with analysis of the image(s)
        """
        if not self.model_info.supports_vision:
            raise ValueError(f"Model {self.model_key} does not support vision")

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        mode = mode or self.default_mode
        messages = self._build_messages(message, mode=mode, images=image_paths)

        params = {
            "model": self.model_id,
            "messages": messages,
            **self._get_mode_params(mode),
            **kwargs
        }

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, mode)

    def chat_with_video(
        self,
        message: str,
        video_path: str,
        mode: Optional[KimiMode] = None,
        **kwargs
    ) -> KimiResponse:
        """
        Chat with video input (experimental, official API only).

        Args:
            message: The user message
            video_path: Path to video file
            mode: Operating mode
            **kwargs: Additional parameters

        Returns:
            KimiResponse with analysis of the video
        """
        if not self.model_info.supports_video:
            raise ValueError(
                f"Model {self.model_key} does not support video. "
                "Video is only supported on official Moonshot API."
            )

        mode = mode or self.default_mode
        video_data = self._encode_video(video_path)

        messages = self._build_messages(message, mode=mode)
        # Modify last message to include video
        messages[-1] = {
            "role": "user",
            "content": [
                {"type": "text", "text": message},
                {"type": "video_url", "video_url": {"url": video_data}},
            ]
        }

        params = {
            "model": self.model_id,
            "messages": messages,
            **self._get_mode_params(mode),
            **kwargs
        }

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, mode)

    def stream(
        self,
        message: str,
        mode: Optional[KimiMode] = None,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[StreamChunk, None, KimiResponse]:
        """
        Stream a chat response.

        Args:
            message: The user message
            mode: Operating mode
            system_prompt: Custom system prompt
            history: Conversation history
            tools: Tool definitions
            **kwargs: Additional parameters

        Yields:
            StreamChunk objects with partial content

        Returns:
            Final KimiResponse (accessible via generator.send(None) or after iteration)
        """
        mode = mode or self.default_mode
        messages = self._build_messages(message, system_prompt, mode, history)

        params = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            **self._get_mode_params(mode),
            **kwargs
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        # Accumulated state
        full_content = ""
        full_reasoning = ""
        tool_calls = []
        finish_reason = None

        stream = self.client.chat.completions.create(**params)

        for chunk in stream:
            delta = chunk.choices[0].delta
            chunk_finish = chunk.choices[0].finish_reason

            stream_chunk = StreamChunk()

            # Handle content
            if delta.content:
                full_content += delta.content
                stream_chunk.content = delta.content

            # Handle reasoning (if present)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                full_reasoning += delta.reasoning_content
                stream_chunk.reasoning = delta.reasoning_content
                stream_chunk.is_reasoning = True

            # Handle tool calls
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    if tc_chunk.index is not None:
                        while len(tool_calls) <= tc_chunk.index:
                            tool_calls.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        tc = tool_calls[tc_chunk.index]
                        if tc_chunk.id:
                            tc["id"] += tc_chunk.id
                        if tc_chunk.function.name:
                            tc["function"]["name"] += tc_chunk.function.name
                        if tc_chunk.function.arguments:
                            tc["function"]["arguments"] += tc_chunk.function.arguments

                stream_chunk.tool_call_delta = delta.tool_calls

            # Capture finish reason
            if chunk_finish:
                finish_reason = chunk_finish
                stream_chunk.finish_reason = chunk_finish

            yield stream_chunk

        # Return final response
        return KimiResponse(
            content=full_content,
            reasoning=full_reasoning if full_reasoning else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason or "stop",
            mode=mode,
            model=self.model_id,
        )

    async def achat(
        self,
        message: str,
        mode: Optional[KimiMode] = None,
        **kwargs
    ) -> KimiResponse:
        """Async version of chat()"""
        mode = mode or self.default_mode
        messages = self._build_messages(message, mode=mode)

        params = {
            "model": self.model_id,
            "messages": messages,
            **self._get_mode_params(mode),
            **kwargs
        }

        response = await self.async_client.chat.completions.create(**params)
        return self._parse_response(response, mode)

    async def astream(
        self,
        message: str,
        mode: Optional[KimiMode] = None,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Async streaming version"""
        mode = mode or self.default_mode
        messages = self._build_messages(message, mode=mode)

        params = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            **self._get_mode_params(mode),
            **kwargs
        }

        stream = await self.async_client.chat.completions.create(**params)

        async for chunk in stream:
            delta = chunk.choices[0].delta
            stream_chunk = StreamChunk()

            if delta.content:
                stream_chunk.content = delta.content
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                stream_chunk.reasoning = delta.reasoning_content
                stream_chunk.is_reasoning = True
            if chunk.choices[0].finish_reason:
                stream_chunk.finish_reason = chunk.choices[0].finish_reason

            yield stream_chunk

    def execute_with_tools(
        self,
        message: str,
        tools: List[Dict[str, Any]],
        tool_map: Dict[str, Callable],
        mode: Optional[KimiMode] = None,
        max_steps: Optional[int] = None,
        on_tool_call: Optional[Callable[[str, Dict], None]] = None,
        on_tool_result: Optional[Callable[[str, Any], None]] = None,
    ) -> KimiResponse:
        """
        Execute a task with automatic tool calling loop.

        This implements the full tool calling pipeline:
        1. Send message with tools
        2. If model requests tool calls, execute them
        3. Feed results back to model
        4. Repeat until completion or max_steps

        Args:
            message: The user task
            tools: Tool definitions
            tool_map: Dict mapping tool names to callable functions
            mode: Operating mode (AGENT or SWARM recommended)
            max_steps: Maximum tool call iterations
            on_tool_call: Callback when tool is called (name, args)
            on_tool_result: Callback when tool returns (name, result)

        Returns:
            Final KimiResponse after tool execution
        """
        mode = mode or KimiMode.AGENT
        max_steps = max_steps or self.tool_config.max_tool_steps

        messages = self._build_messages(message, mode=mode)
        step_count = 0

        while step_count < max_steps:
            # Make API request
            params = {
                "model": self.model_id,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                **self._get_mode_params(mode),
            }

            response = self.client.chat.completions.create(**params)
            choice = response.choices[0]

            # Log reasoning if present
            if (self.tool_config.log_reasoning and
                hasattr(choice.message, "reasoning_content") and
                choice.message.reasoning_content):
                logger.info(f"Reasoning: {choice.message.reasoning_content[:200]}...")

            # Check if done
            if choice.finish_reason != "tool_calls":
                return self._parse_response(response, mode)

            # Append assistant message
            messages.append(choice.message)

            # Execute tool calls
            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if on_tool_call:
                    on_tool_call(tool_name, tool_args)

                # Execute tool
                if tool_name not in tool_map:
                    result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        result = tool_map[tool_name](**tool_args)
                    except Exception as e:
                        result = {"error": str(e)}

                if on_tool_result:
                    on_tool_result(tool_name, result)

                # Append tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(result) if not isinstance(result, str) else result,
                })

                step_count += 1

        # Max steps reached
        logger.warning(f"Reached max tool steps ({max_steps})")
        return self._parse_response(response, mode)


# Convenience function for quick usage
def create_client(
    mode: KimiMode = DEFAULT_MODE,
    **kwargs
) -> KimiClient:
    """Create a KimiClient with specified default mode"""
    return KimiClient(default_mode=mode, **kwargs)


# Quick access functions
def instant(message: str, **kwargs) -> KimiResponse:
    """Quick instant mode response"""
    client = KimiClient(default_mode=KimiMode.INSTANT)
    return client.chat(message, **kwargs)


def think(message: str, **kwargs) -> KimiResponse:
    """Quick thinking mode response with reasoning"""
    client = KimiClient(default_mode=KimiMode.THINKING)
    return client.chat(message, **kwargs)


if __name__ == "__main__":
    # Demo usage
    print("Kimi K2.5 Client Demo")
    print("=" * 50)

    client = KimiClient()

    # Test instant mode
    print("\n[Instant Mode]")
    response = client.chat("What is 2+2?", mode=KimiMode.INSTANT)
    print(f"Response: {response.content}")

    # Test thinking mode
    print("\n[Thinking Mode]")
    response = client.chat(
        "What's larger: 9.11 or 9.9? Think carefully.",
        mode=KimiMode.THINKING
    )
    if response.reasoning:
        print(f"Reasoning: {response.reasoning[:200]}...")
    print(f"Response: {response.content}")
