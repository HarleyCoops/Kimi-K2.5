#!/usr/bin/env python3
"""
Kimi K2.5 MCP Bridge

Connects Kimi K2.5's tool calling system to MCP (Model Context Protocol) servers.
This enables the swarm orchestrator to use real tools from:
- ModelScope MCP Server (image generation, model search, etc.)
- Custom MCP servers (any stdio or HTTP-based MCP server)

The bridge handles:
1. MCP server connection (stdio/HTTP/SSE transports)
2. Tool discovery and schema conversion
3. Tool execution and result formatting
4. Async support for parallel tool calls

Example:
    from tools.mcp_bridge import MCPBridge

    # Connect to ModelScope MCP server
    bridge = MCPBridge()
    await bridge.connect_stdio("modelscope", "uvx", ["modelscope-mcp-server"])

    # Get tools for Kimi
    tools = bridge.get_openai_tools()
    tool_map = bridge.get_tool_map()

    # Use with swarm orchestrator
    orchestrator = SwarmOrchestrator(tools=tools, tool_map=tool_map)
"""

import os
import json
import asyncio
import subprocess
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import requests

logger = logging.getLogger(__name__)


@dataclass
class MCPServer:
    """Represents a connected MCP server"""
    name: str
    transport: str  # "stdio", "http", "sse"
    process: Optional[subprocess.Popen] = None
    url: Optional[str] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    connected: bool = False


@dataclass
class MCPToolResult:
    """Result from an MCP tool call"""
    success: bool
    content: Any
    error: Optional[str] = None


class MCPBridge:
    """
    Bridge between Kimi K2.5 and MCP servers.

    Supports multiple MCP servers simultaneously, each providing
    different tool capabilities.
    """

    def __init__(self):
        """Initialize the MCP bridge"""
        self.servers: Dict[str, MCPServer] = {}
        self._request_id = 0

    def _next_request_id(self) -> int:
        """Generate unique request ID"""
        self._request_id += 1
        return self._request_id

    async def connect_stdio(
        self,
        name: str,
        command: str,
        args: List[str] = None,
        env: Dict[str, str] = None,
    ) -> bool:
        """
        Connect to an MCP server via stdio transport.

        Args:
            name: Server name for reference
            command: Command to launch (e.g., "uvx", "npx", "python")
            args: Command arguments
            env: Environment variables

        Returns:
            True if connected successfully
        """
        try:
            # Merge environment
            full_env = os.environ.copy()
            if env:
                full_env.update(env)

            # Start the MCP server process
            process = subprocess.Popen(
                [command] + (args or []),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=full_env,
                text=True,
                bufsize=1,
            )

            server = MCPServer(
                name=name,
                transport="stdio",
                process=process,
                connected=True,
            )

            # Initialize and discover tools
            await self._initialize_stdio(server)

            self.servers[name] = server
            logger.info(f"Connected to MCP server '{name}' with {len(server.tools)} tools")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{name}': {e}")
            return False

    async def connect_http(
        self,
        name: str,
        url: str,
        headers: Dict[str, str] = None,
    ) -> bool:
        """
        Connect to an MCP server via HTTP/SSE transport.

        Args:
            name: Server name for reference
            url: Server URL (e.g., "http://127.0.0.1:8000/mcp/")
            headers: Optional HTTP headers (for auth)

        Returns:
            True if connected successfully
        """
        try:
            server = MCPServer(
                name=name,
                transport="http",
                url=url,
                connected=True,
            )

            # Discover tools via HTTP
            await self._initialize_http(server, headers)

            self.servers[name] = server
            logger.info(f"Connected to MCP server '{name}' at {url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to HTTP MCP server '{name}': {e}")
            return False

    async def _initialize_stdio(self, server: MCPServer):
        """Initialize stdio server and discover tools"""
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "kimi-k2.5-swarm",
                    "version": "1.0.0"
                }
            }
        }

        response = await self._send_stdio_request(server, init_request)

        # Send initialized notification
        initialized = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        server.process.stdin.write(json.dumps(initialized) + "\n")
        server.process.stdin.flush()

        # List tools
        tools_request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/list",
            "params": {}
        }

        tools_response = await self._send_stdio_request(server, tools_request)
        if tools_response and "result" in tools_response:
            server.tools = tools_response["result"].get("tools", [])

    async def _initialize_http(self, server: MCPServer, headers: Dict[str, str] = None):
        """Initialize HTTP server and discover tools"""
        async with aiohttp.ClientSession() as session:
            # Initialize
            init_payload = {
                "jsonrpc": "2.0",
                "id": self._next_request_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "kimi-k2.5-swarm",
                        "version": "1.0.0"
                    }
                }
            }

            async with session.post(
                server.url,
                json=init_payload,
                headers=headers
            ) as resp:
                await resp.json()

            # List tools
            tools_payload = {
                "jsonrpc": "2.0",
                "id": self._next_request_id(),
                "method": "tools/list",
                "params": {}
            }

            async with session.post(
                server.url,
                json=tools_payload,
                headers=headers
            ) as resp:
                result = await resp.json()
                if "result" in result:
                    server.tools = result["result"].get("tools", [])

    async def _send_stdio_request(
        self,
        server: MCPServer,
        request: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Send a request via stdio and wait for response"""
        try:
            server.process.stdin.write(json.dumps(request) + "\n")
            server.process.stdin.flush()

            # Read response (with timeout)
            loop = asyncio.get_event_loop()
            response_line = await asyncio.wait_for(
                loop.run_in_executor(None, server.process.stdout.readline),
                timeout=30.0
            )

            if response_line:
                return json.loads(response_line)
            return None

        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for response from {server.name}")
            return None
        except Exception as e:
            logger.error(f"Error communicating with {server.name}: {e}")
            return None

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> MCPToolResult:
        """
        Call a tool on an MCP server.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            MCPToolResult with the tool output
        """
        server = self.servers.get(server_name)
        if not server or not server.connected:
            return MCPToolResult(
                success=False,
                content=None,
                error=f"Server '{server_name}' not connected"
            )

        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        try:
            if server.transport == "stdio":
                response = await self._send_stdio_request(server, request)
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.post(server.url, json=request) as resp:
                        response = await resp.json()

            if response and "result" in response:
                content = response["result"].get("content", [])
                # Extract text content
                text_content = []
                for item in content:
                    if item.get("type") == "text":
                        text_content.append(item.get("text", ""))

                return MCPToolResult(
                    success=True,
                    content="\n".join(text_content) if text_content else content
                )
            elif response and "error" in response:
                return MCPToolResult(
                    success=False,
                    content=None,
                    error=response["error"].get("message", "Unknown error")
                )
            else:
                return MCPToolResult(
                    success=False,
                    content=None,
                    error="No response from server"
                )

        except Exception as e:
            return MCPToolResult(
                success=False,
                content=None,
                error=str(e)
            )

    def get_openai_tools(self, server_name: str = None) -> List[Dict[str, Any]]:
        """
        Get tools in OpenAI function calling format.

        Args:
            server_name: Specific server, or None for all servers

        Returns:
            List of tool schemas
        """
        tools = []

        servers = [self.servers[server_name]] if server_name else self.servers.values()

        for server in servers:
            for mcp_tool in server.tools:
                # Convert MCP tool schema to OpenAI format
                tool = {
                    "type": "function",
                    "function": {
                        "name": f"{server.name}__{mcp_tool['name']}",
                        "description": mcp_tool.get("description", ""),
                        "parameters": mcp_tool.get("inputSchema", {
                            "type": "object",
                            "properties": {},
                            "required": []
                        })
                    }
                }
                tools.append(tool)

        return tools

    def get_tool_map(self, server_name: str = None) -> Dict[str, Callable]:
        """
        Get a mapping of tool names to async callables.

        The returned callables can be used directly with Kimi's
        tool execution system.

        Args:
            server_name: Specific server, or None for all servers

        Returns:
            Dict mapping tool names to async functions
        """
        tool_map = {}

        servers = [self.servers[server_name]] if server_name else self.servers.values()

        for server in servers:
            for mcp_tool in server.tools:
                full_name = f"{server.name}__{mcp_tool['name']}"

                # Create a closure to capture server and tool names
                def make_caller(srv_name: str, tl_name: str):
                    async def call_mcp_tool(**kwargs) -> Dict[str, Any]:
                        result = await self.call_tool(srv_name, tl_name, kwargs)
                        if result.success:
                            return {"result": result.content}
                        else:
                            return {"error": result.error}
                    return call_mcp_tool

                tool_map[full_name] = make_caller(server.name, mcp_tool["name"])

        return tool_map

    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools across all servers"""
        tools = []
        for server in self.servers.values():
            for tool in server.tools:
                tools.append({
                    "server": server.name,
                    "name": tool["name"],
                    "description": tool.get("description", "")[:100]
                })
        return tools

    async def disconnect(self, server_name: str = None):
        """Disconnect from MCP server(s)"""
        servers = [server_name] if server_name else list(self.servers.keys())

        for name in servers:
            server = self.servers.get(name)
            if server:
                if server.process:
                    server.process.terminate()
                    server.process.wait(timeout=5)
                server.connected = False
                del self.servers[name]
                logger.info(f"Disconnected from MCP server '{name}'")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


def load_mcp_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load MCP configuration from standard config file.

    Searches for config in order:
    1. Provided path
    2. ./.mcp.json
    3. ~/.config/kimi/mcp.json
    4. ~/.claude/claude_desktop_config.json

    Returns:
        Dict with mcpServers configuration
    """
    search_paths = [
        config_path,
        ".mcp.json",
        Path.home() / ".config" / "kimi" / "mcp.json",
        Path.home() / ".claude" / "claude_desktop_config.json",
    ]

    for path in search_paths:
        if path and Path(path).exists():
            with open(path) as f:
                config = json.load(f)
                if "mcpServers" in config:
                    return config

    return {"mcpServers": {}}


async def create_bridge_from_config(config_path: str = None) -> MCPBridge:
    """
    Create and connect an MCP bridge from config file.

    Args:
        config_path: Path to MCP config file

    Returns:
        Connected MCPBridge instance
    """
    config = load_mcp_config(config_path)
    bridge = MCPBridge()

    for name, server_config in config.get("mcpServers", {}).items():
        if "url" in server_config:
            # HTTP transport
            await bridge.connect_http(
                name=name,
                url=server_config["url"],
                headers=server_config.get("headers")
            )
        elif "command" in server_config:
            # Stdio transport
            await bridge.connect_stdio(
                name=name,
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env", {})
            )

    return bridge


if __name__ == "__main__":
    print("Kimi K2.5 MCP Bridge")
    print("=" * 50)
    print("\nUsage:")
    print("  bridge = MCPBridge()")
    print("  await bridge.connect_stdio('modelscope', 'uvx', ['modelscope-mcp-server'])")
    print("  tools = bridge.get_openai_tools()")
    print("  tool_map = bridge.get_tool_map()")
    print("\nOr from config:")
    print("  bridge = await create_bridge_from_config('.mcp.json')")
