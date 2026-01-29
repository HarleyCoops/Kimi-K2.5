"""
Kimi K2.5 Tools Module

Enhanced tool calling with support for:
- Thinking mode interleaved reasoning (200-300 step chains)
- Parallel execution (up to 1,500 concurrent calls)
- Tool registry and dynamic loading
- Built-in tools for common operations
- MCP (Model Context Protocol) server integration

Features:
- ThinkingToolExecutor: Long chains with reasoning traces
- ParallelToolExecutor: Concurrent execution with rate limiting
- ToolRegistry: Centralized tool management
- MCPBridge: Connect to MCP servers for real tools
- Built-in tools: Web search, file ops, code execution, API requests

Example:
    from tools import ParallelToolExecutor, ToolRegistry, MCPBridge

    # Use MCP servers for real tools
    bridge = MCPBridge()
    await bridge.connect_stdio("modelscope", "uvx", ["modelscope-mcp-server"])
    tools = bridge.get_openai_tools()
    tool_map = bridge.get_tool_map()

    # Or use the registry for custom tools
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
from .mcp_bridge import MCPBridge, create_bridge_from_config, load_mcp_config

__all__ = [
    "ParallelToolExecutor",
    "ToolRegistry",
    "Tool",
    "ThinkingToolExecutor",
    "MCPBridge",
    "create_bridge_from_config",
    "load_mcp_config",
    "web_search",
    "read_url",
    "calculate",
    "get_weather",
    "BUILTIN_TOOLS",
]
