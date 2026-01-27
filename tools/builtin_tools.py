#!/usr/bin/env python3
"""
Kimi K2.5 Built-in Tools

Pre-configured tools for common operations:
- web_search: Search the web
- read_url: Fetch content from URLs
- calculate: Mathematical calculations
- get_weather: Weather information
- file_read/write: File operations
- run_code: Code execution (sandboxed)

These tools are ready to use with Kimi K2.5's tool calling capabilities.

Example:
    from tools.builtin_tools import BUILTIN_TOOLS, BUILTIN_TOOL_MAP

    # Use with KimiClient
    response = client.chat(
        "Search for latest AI news",
        tools=BUILTIN_TOOLS,
        tool_choice="auto"
    )
"""

import json
import math
import re
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Implementations
# ============================================================================

def calculate(expression: str) -> Dict[str, Any]:
    """
    Evaluate a mathematical expression safely.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Dict with result or error
    """
    # Whitelist safe characters and functions
    allowed_chars = set('0123456789+-*/%().^ ')
    allowed_funcs = {
        'sin', 'cos', 'tan', 'sqrt', 'log', 'log10', 'exp',
        'abs', 'round', 'floor', 'ceil', 'pow', 'pi', 'e'
    }

    # Clean expression
    expr = expression.strip()

    # Check for disallowed characters (excluding function names)
    expr_check = expr
    for func in allowed_funcs:
        expr_check = expr_check.replace(func, '')

    if not all(c in allowed_chars or c.isalpha() for c in expr_check):
        return {"error": "Invalid characters in expression"}

    try:
        # Replace common operators
        expr = expr.replace('^', '**')

        # Create safe namespace with math functions
        safe_dict = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
            'exp': math.exp, 'abs': abs, 'round': round,
            'floor': math.floor, 'ceil': math.ceil, 'pow': pow,
            'pi': math.pi, 'e': math.e
        }

        result = eval(expr, {"__builtins__": {}}, safe_dict)
        return {"result": result, "expression": expression}

    except Exception as e:
        return {"error": f"Calculation failed: {str(e)}"}


def get_weather(city: str) -> Dict[str, Any]:
    """
    Get weather information for a city (mock implementation).

    Args:
        city: Name of the city

    Returns:
        Dict with weather data
    """
    # Mock weather data - in production, call a real weather API
    mock_data = {
        "Beijing": {"weather": "Sunny", "temperature": "22°C", "humidity": "45%", "wind": "5 km/h NE"},
        "Shanghai": {"weather": "Cloudy", "temperature": "18°C", "humidity": "60%", "wind": "8 km/h E"},
        "New York": {"weather": "Rainy", "temperature": "15°C", "humidity": "80%", "wind": "12 km/h SW"},
        "London": {"weather": "Foggy", "temperature": "12°C", "humidity": "75%", "wind": "6 km/h W"},
        "Tokyo": {"weather": "Partly Cloudy", "temperature": "20°C", "humidity": "55%", "wind": "4 km/h N"},
        "Paris": {"weather": "Clear", "temperature": "17°C", "humidity": "50%", "wind": "7 km/h S"},
        "Sydney": {"weather": "Sunny", "temperature": "25°C", "humidity": "40%", "wind": "10 km/h SE"},
    }

    # Case-insensitive lookup
    city_lower = city.lower()
    for key, value in mock_data.items():
        if key.lower() == city_lower:
            return {"city": key, **value}

    return {
        "city": city,
        "weather": "Unknown",
        "note": "City not in database. This is mock data."
    }


def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Search the web for information (mock implementation).

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        Dict with search results
    """
    # Mock implementation - in production, use a real search API
    logger.info(f"Web search query: {query}")

    return {
        "query": query,
        "results": [
            {
                "title": f"Result {i+1} for: {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a mock search result for '{query}'. In production, this would contain real search results."
            }
            for i in range(min(num_results, 5))
        ],
        "note": "This is mock data. Implement real search API for production use."
    }


def read_url(url: str) -> Dict[str, Any]:
    """
    Read content from a URL (mock implementation).

    Args:
        url: URL to read

    Returns:
        Dict with URL content
    """
    logger.info(f"Reading URL: {url}")

    # Mock implementation
    return {
        "url": url,
        "content": f"Mock content from {url}. In production, this would fetch the actual page content.",
        "status": 200,
        "note": "This is mock data. Implement real HTTP client for production use."
    }


def run_code(code: str, language: str = "python") -> Dict[str, Any]:
    """
    Execute code (mock/sandboxed implementation).

    Args:
        code: Code to execute
        language: Programming language

    Returns:
        Dict with execution result
    """
    logger.warning(f"Code execution requested (language: {language})")

    # This is a mock - actual implementation should use a sandboxed environment
    return {
        "language": language,
        "code": code[:500] + ("..." if len(code) > 500 else ""),
        "output": "Code execution is disabled in this mock implementation.",
        "status": "mock",
        "note": "Implement sandboxed execution (e.g., Docker, subprocess) for production."
    }


def file_read(path: str) -> Dict[str, Any]:
    """
    Read file contents (mock implementation).

    Args:
        path: File path to read

    Returns:
        Dict with file content
    """
    logger.info(f"File read requested: {path}")

    return {
        "path": path,
        "content": f"Mock file content for {path}",
        "status": "mock",
        "note": "Implement with proper security controls for production."
    }


def file_write(path: str, content: str) -> Dict[str, Any]:
    """
    Write content to file (mock implementation).

    Args:
        path: File path to write
        content: Content to write

    Returns:
        Dict with operation result
    """
    logger.info(f"File write requested: {path}")

    return {
        "path": path,
        "bytes_written": len(content),
        "status": "mock",
        "note": "Implement with proper security controls for production."
    }


# ============================================================================
# Tool Schemas (OpenAI format)
# ============================================================================

CALCULATE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Supports basic arithmetic, trigonometry (sin, cos, tan), logarithms (log, log10), and constants (pi, e).",
        "parameters": {
            "type": "object",
            "required": ["expression"],
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2 * 3', 'sqrt(16)', 'sin(pi/2)')"
                }
            }
        }
    }
}

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather information for a city including temperature, conditions, humidity, and wind.",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Name of the city (e.g., 'Beijing', 'New York', 'London')"
                }
            }
        }
    }
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information. Returns relevant results with titles, URLs, and snippets.",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 10)",
                    "default": 5
                }
            }
        }
    }
}

READ_URL_TOOL = {
    "type": "function",
    "function": {
        "name": "read_url",
        "description": "Fetch and read content from a URL. Useful for reading web pages, APIs, or documents.",
        "parameters": {
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to read"
                }
            }
        }
    }
}

RUN_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "run_code",
        "description": "Execute code in a sandboxed environment. Supports Python and JavaScript.",
        "parameters": {
            "type": "object",
            "required": ["code"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Code to execute"
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript"],
                    "description": "Programming language",
                    "default": "python"
                }
            }
        }
    }
}

FILE_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "file_read",
        "description": "Read contents of a file.",
        "parameters": {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            }
        }
    }
}

FILE_WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "file_write",
        "description": "Write content to a file.",
        "parameters": {
            "type": "object",
            "required": ["path", "content"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to write the file"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write"
                }
            }
        }
    }
}


# ============================================================================
# Exports
# ============================================================================

# All built-in tools
BUILTIN_TOOLS = [
    CALCULATE_TOOL,
    GET_WEATHER_TOOL,
    WEB_SEARCH_TOOL,
    READ_URL_TOOL,
    RUN_CODE_TOOL,
    FILE_READ_TOOL,
    FILE_WRITE_TOOL,
]

# Common tools (most frequently used)
COMMON_TOOLS = [
    CALCULATE_TOOL,
    WEB_SEARCH_TOOL,
    READ_URL_TOOL,
]

# Tool function mapping
BUILTIN_TOOL_MAP = {
    "calculate": calculate,
    "get_weather": get_weather,
    "web_search": web_search,
    "read_url": read_url,
    "run_code": run_code,
    "file_read": file_read,
    "file_write": file_write,
}


if __name__ == "__main__":
    print("Kimi K2.5 Built-in Tools")
    print("=" * 50)
    print("\nAvailable tools:")
    for tool in BUILTIN_TOOLS:
        name = tool["function"]["name"]
        desc = tool["function"]["description"][:60] + "..."
        print(f"  - {name}: {desc}")

    print("\nTest calculations:")
    print(f"  2 + 2 * 3 = {calculate('2 + 2 * 3')}")
    print(f"  sqrt(16) = {calculate('sqrt(16)')}")
    print(f"  sin(pi/2) = {calculate('sin(pi/2)')}")
