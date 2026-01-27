"""
Kimi K2.5 Tools Module

Enhanced tool calling with support for:
- Thinking mode interleaved reasoning (200-300 step chains)
- Parallel execution (up to 1,500 concurrent calls)
- Tool registry and dynamic loading
- Built-in tools for common operations

Features:
- ThinkingToolExecutor: Long chains with reasoning traces
- ParallelToolExecutor: Concurrent execution with rate limiting
- ToolRegistry: Centralized tool management
- Built-in tools: Web search, file ops, code execution, API requests

Example:
    from tools import ParallelToolExecutor, ToolRegistry

    registry = ToolRegistry()
    registry.register("web_search", web_search_func, schema)

    executor = ParallelToolExecutor(max_concurrent=100)
    results = await executor.execute_batch(tool_calls, registry)
"""

from .parallel_executor import ParallelToolExecutor
from .tool_registry import ToolRegistry, Tool
from .builtin_tools import (
    web_search,
    read_url,
    calculate,
    get_weather,
    BUILTIN_TOOLS,
)
from .thinking_tools import ThinkingToolExecutor

__all__ = [
    "ParallelToolExecutor",
    "ToolRegistry",
    "Tool",
    "ThinkingToolExecutor",
    "web_search",
    "read_url",
    "calculate",
    "get_weather",
    "BUILTIN_TOOLS",
]
