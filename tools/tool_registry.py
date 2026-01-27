#!/usr/bin/env python3
"""
Kimi K2.5 Tool Registry

Centralized tool management for:
- Tool registration and discovery
- Schema validation
- Dynamic tool loading
- Tool categorization and tagging

Example:
    from tools import ToolRegistry

    registry = ToolRegistry()

    # Register a tool
    registry.register(
        name="calculate",
        func=calculate_func,
        description="Perform calculations",
        parameters={
            "type": "object",
            "required": ["expression"],
            "properties": {
                "expression": {"type": "string"}
            }
        }
    )

    # Get tools for API
    tools = registry.get_openai_tools()
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import inspect
import logging

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Representation of a registered tool"""
    name: str
    func: Callable
    description: str
    parameters: Dict[str, Any]
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    enabled: bool = True
    requires_auth: bool = False

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def __call__(self, *args, **kwargs):
        """Make the tool callable"""
        return self.func(*args, **kwargs)


class ToolRegistry:
    """
    Registry for managing tools available to Kimi K2.5.

    Features:
    - Register tools with schemas
    - Get tools by category/tags
    - Enable/disable tools dynamically
    - Export to OpenAI format
    """

    def __init__(self):
        """Initialize empty registry"""
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(
        self,
        name: str,
        func: Callable,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        category: str = "general",
        tags: Optional[List[str]] = None,
        requires_auth: bool = False,
    ) -> Tool:
        """
        Register a new tool.

        Args:
            name: Tool name (must be unique)
            func: Callable function
            description: Tool description (auto-generated from docstring if not provided)
            parameters: JSON schema for parameters
            category: Tool category for grouping
            tags: Optional tags for filtering
            requires_auth: Whether tool requires authentication

        Returns:
            Registered Tool object
        """
        # Auto-generate description from docstring
        if description is None:
            description = func.__doc__ or f"Execute {name}"
            description = description.strip().split("\n")[0]

        # Auto-generate parameters from signature
        if parameters is None:
            parameters = self._infer_parameters(func)

        tool = Tool(
            name=name,
            func=func,
            description=description,
            parameters=parameters,
            category=category,
            tags=tags or [],
            requires_auth=requires_auth,
        )

        self._tools[name] = tool

        # Update category index
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)

        logger.info(f"Registered tool: {name} (category: {category})")
        return tool

    def _infer_parameters(self, func: Callable) -> Dict[str, Any]:
        """Infer parameters schema from function signature"""
        sig = inspect.signature(func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Determine type
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"

            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter: {param_name}"
            }

            # Check if required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def get_func(self, name: str) -> Optional[Callable]:
        """Get a tool's function by name"""
        tool = self._tools.get(name)
        return tool.func if tool else None

    def remove(self, name: str) -> bool:
        """Remove a tool from registry"""
        if name in self._tools:
            tool = self._tools.pop(name)
            if tool.category in self._categories:
                self._categories[tool.category].remove(name)
            logger.info(f"Removed tool: {name}")
            return True
        return False

    def enable(self, name: str) -> bool:
        """Enable a tool"""
        if name in self._tools:
            self._tools[name].enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a tool"""
        if name in self._tools:
            self._tools[name].enabled = False
            return True
        return False

    def list_tools(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        enabled_only: bool = True,
    ) -> List[str]:
        """
        List tool names with optional filtering.

        Args:
            category: Filter by category
            tags: Filter by tags (any match)
            enabled_only: Only return enabled tools

        Returns:
            List of tool names
        """
        tools = list(self._tools.values())

        if enabled_only:
            tools = [t for t in tools if t.enabled]

        if category:
            tools = [t for t in tools if t.category == category]

        if tags:
            tools = [t for t in tools if any(tag in t.tags for tag in tags)]

        return [t.name for t in tools]

    def get_categories(self) -> List[str]:
        """Get all registered categories"""
        return list(self._categories.keys())

    def get_openai_tools(
        self,
        category: Optional[str] = None,
        names: Optional[List[str]] = None,
        enabled_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get tools in OpenAI function calling format.

        Args:
            category: Filter by category
            names: Only include these tools
            enabled_only: Only enabled tools

        Returns:
            List of tool schemas for OpenAI API
        """
        tools = []

        for name, tool in self._tools.items():
            # Apply filters
            if enabled_only and not tool.enabled:
                continue
            if category and tool.category != category:
                continue
            if names and name not in names:
                continue

            tools.append(tool.to_openai_schema())

        return tools

    def get_tool_map(
        self,
        names: Optional[List[str]] = None,
        enabled_only: bool = True,
    ) -> Dict[str, Callable]:
        """
        Get mapping of tool names to functions.

        Args:
            names: Only include these tools
            enabled_only: Only enabled tools

        Returns:
            Dict mapping tool names to callable functions
        """
        result = {}

        for name, tool in self._tools.items():
            if enabled_only and not tool.enabled:
                continue
            if names and name not in names:
                continue
            result[name] = tool.func

        return result

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Global default registry
_default_registry: Optional[ToolRegistry] = None


def get_default_registry() -> ToolRegistry:
    """Get or create the default global registry"""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def register_tool(
    name: str,
    func: Callable,
    **kwargs
) -> Tool:
    """Register a tool in the default registry"""
    return get_default_registry().register(name, func, **kwargs)


if __name__ == "__main__":
    print("Kimi K2.5 Tool Registry")
    print("=" * 50)
    print("\nUsage:")
    print("  registry = ToolRegistry()")
    print("  registry.register('my_tool', my_func, 'Description')")
    print("  tools = registry.get_openai_tools()")
    print("  tool_map = registry.get_tool_map()")
