#!/usr/bin/env python3
"""
Kimi K2.5 Browser Agent

AI-controlled browser automation using Playwright.
The agent can autonomously:
- Navigate to URLs
- Search the web
- Click elements
- Fill forms
- Extract content
- Take screenshots

This enables real-world research, data collection, and web interaction
powered by Kimi K2.5's reasoning capabilities.

Example:
    from live_demos import BrowserAgent

    async with BrowserAgent() as agent:
        # Let K2.5 research a topic
        result = await agent.research("Latest developments in quantum computing")
        print(result.summary)
        print(f"Sources: {result.sources}")

        # Or manually control
        await agent.navigate("https://example.com")
        content = await agent.extract_content()
"""

import asyncio
import base64
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging

try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = Page = BrowserContext = None

from kimi_client import KimiClient, KimiMode, KimiResponse
from tools import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class BrowserState:
    """Current state of the browser"""
    url: str = ""
    title: str = ""
    content_snippet: str = ""
    screenshot_path: Optional[str] = None


@dataclass
class ResearchResult:
    """Result from browser-based research"""
    query: str
    summary: str
    sources: List[Dict[str, str]]  # [{url, title, snippet}]
    screenshots: List[str]
    tool_calls_made: int
    success: bool


class BrowserTools:
    """
    Browser tools for Kimi K2.5 to control Playwright.

    Provides tool schemas and implementations for:
    - Navigation
    - Clicking
    - Form filling
    - Content extraction
    - Screenshot capture
    """

    def __init__(self, page: Optional[Page] = None):
        self.page = page
        self._screenshot_count = 0

    def set_page(self, page: Page):
        """Set the Playwright page object"""
        self.page = page

    # Tool implementations

    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL"""
        if not self.page:
            return {"error": "Browser not initialized"}

        try:
            await self.page.goto(url, wait_until="networkidle", timeout=30000)
            return {
                "url": self.page.url,
                "title": await self.page.title(),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e)}

    async def click(self, selector: str) -> Dict[str, Any]:
        """Click an element by selector"""
        if not self.page:
            return {"error": "Browser not initialized"}

        try:
            await self.page.click(selector, timeout=5000)
            await self.page.wait_for_load_state("networkidle", timeout=10000)
            return {"status": "clicked", "selector": selector}
        except Exception as e:
            return {"error": str(e)}

    async def fill(self, selector: str, value: str) -> Dict[str, Any]:
        """Fill a form field"""
        if not self.page:
            return {"error": "Browser not initialized"}

        try:
            await self.page.fill(selector, value)
            return {"status": "filled", "selector": selector, "value": value}
        except Exception as e:
            return {"error": str(e)}

    async def type_text(self, selector: str, text: str) -> Dict[str, Any]:
        """Type text character by character (useful for search boxes)"""
        if not self.page:
            return {"error": "Browser not initialized"}

        try:
            await self.page.type(selector, text, delay=50)
            return {"status": "typed", "selector": selector}
        except Exception as e:
            return {"error": str(e)}

    async def press_key(self, key: str) -> Dict[str, Any]:
        """Press a keyboard key"""
        if not self.page:
            return {"error": "Browser not initialized"}

        try:
            await self.page.keyboard.press(key)
            return {"status": "pressed", "key": key}
        except Exception as e:
            return {"error": str(e)}

    async def extract_text(self, selector: Optional[str] = None) -> Dict[str, Any]:
        """Extract text content from page or element"""
        if not self.page:
            return {"error": "Browser not initialized"}

        try:
            if selector:
                element = await self.page.query_selector(selector)
                if element:
                    text = await element.text_content()
                else:
                    return {"error": f"Element not found: {selector}"}
            else:
                text = await self.page.text_content("body")

            # Truncate for context
            text = text[:10000] if text else ""
            return {"text": text, "length": len(text)}
        except Exception as e:
            return {"error": str(e)}

    async def get_links(self) -> Dict[str, Any]:
        """Get all links on the page"""
        if not self.page:
            return {"error": "Browser not initialized"}

        try:
            links = await self.page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]'))
                    .slice(0, 50)
                    .map(a => ({
                        text: a.textContent?.trim().slice(0, 100) || '',
                        href: a.href
                    }))
                    .filter(l => l.text && l.href.startsWith('http'))
            """)
            return {"links": links, "count": len(links)}
        except Exception as e:
            return {"error": str(e)}

    async def screenshot(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Take a screenshot"""
        if not self.page:
            return {"error": "Browser not initialized"}

        try:
            self._screenshot_count += 1
            if not path:
                path = f"screenshot_{self._screenshot_count}.png"

            await self.page.screenshot(path=path, full_page=False)
            return {"path": path, "status": "captured"}
        except Exception as e:
            return {"error": str(e)}

    async def scroll(self, direction: str = "down", amount: int = 500) -> Dict[str, Any]:
        """Scroll the page"""
        if not self.page:
            return {"error": "Browser not initialized"}

        try:
            if direction == "down":
                await self.page.evaluate(f"window.scrollBy(0, {amount})")
            elif direction == "up":
                await self.page.evaluate(f"window.scrollBy(0, -{amount})")
            return {"status": "scrolled", "direction": direction, "amount": amount}
        except Exception as e:
            return {"error": str(e)}

    async def wait(self, seconds: float) -> Dict[str, Any]:
        """Wait for a specified time"""
        await asyncio.sleep(seconds)
        return {"status": "waited", "seconds": seconds}

    async def get_page_info(self) -> Dict[str, Any]:
        """Get current page information"""
        if not self.page:
            return {"error": "Browser not initialized"}

        try:
            return {
                "url": self.page.url,
                "title": await self.page.title(),
            }
        except Exception as e:
            return {"error": str(e)}

    # Tool schemas for OpenAI format

    @staticmethod
    def get_tool_schemas() -> List[Dict[str, Any]]:
        """Get all browser tool schemas"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "browser_navigate",
                    "description": "Navigate the browser to a URL",
                    "parameters": {
                        "type": "object",
                        "required": ["url"],
                        "properties": {
                            "url": {"type": "string", "description": "URL to navigate to"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_click",
                    "description": "Click an element on the page using CSS selector",
                    "parameters": {
                        "type": "object",
                        "required": ["selector"],
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector of element to click"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_fill",
                    "description": "Fill a form field with text",
                    "parameters": {
                        "type": "object",
                        "required": ["selector", "value"],
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector of input field"},
                            "value": {"type": "string", "description": "Text to fill"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_type",
                    "description": "Type text character by character (for search boxes)",
                    "parameters": {
                        "type": "object",
                        "required": ["selector", "text"],
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector"},
                            "text": {"type": "string", "description": "Text to type"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_press_key",
                    "description": "Press a keyboard key (Enter, Tab, Escape, etc.)",
                    "parameters": {
                        "type": "object",
                        "required": ["key"],
                        "properties": {
                            "key": {"type": "string", "description": "Key to press (e.g., 'Enter', 'Tab')"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_extract_text",
                    "description": "Extract text content from the page or a specific element",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "Optional CSS selector (extracts whole page if not provided)"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_get_links",
                    "description": "Get all links on the current page",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_screenshot",
                    "description": "Take a screenshot of the current page",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to save screenshot (optional)"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_scroll",
                    "description": "Scroll the page up or down",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "direction": {"type": "string", "enum": ["up", "down"], "default": "down"},
                            "amount": {"type": "integer", "default": 500}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_wait",
                    "description": "Wait for a specified number of seconds",
                    "parameters": {
                        "type": "object",
                        "required": ["seconds"],
                        "properties": {
                            "seconds": {"type": "number", "description": "Seconds to wait"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_page_info",
                    "description": "Get current page URL and title",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
        ]

    def get_tool_map(self) -> Dict[str, Callable]:
        """Get mapping of tool names to implementations"""
        return {
            "browser_navigate": self.navigate,
            "browser_click": self.click,
            "browser_fill": self.fill,
            "browser_type": self.type_text,
            "browser_press_key": self.press_key,
            "browser_extract_text": self.extract_text,
            "browser_get_links": self.get_links,
            "browser_screenshot": self.screenshot,
            "browser_scroll": self.scroll,
            "browser_wait": self.wait,
            "browser_page_info": self.get_page_info,
        }


class BrowserAgent:
    """
    Kimi K2.5 powered browser automation agent.

    The agent uses K2.5's reasoning capabilities to:
    - Plan navigation strategies
    - Decide what to click/extract
    - Conduct research autonomously
    - Handle errors and adapt

    Usage:
        async with BrowserAgent() as agent:
            result = await agent.research("quantum computing")
    """

    def __init__(
        self,
        client: Optional[KimiClient] = None,
        headless: bool = True,
        screenshot_dir: str = "./screenshots",
    ):
        """
        Initialize the browser agent.

        Args:
            client: KimiClient instance
            headless: Run browser in headless mode
            screenshot_dir: Directory for screenshots
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright not installed. Install with: pip install playwright && playwright install"
            )

        self.client = client or KimiClient(default_mode=KimiMode.AGENT)
        self.headless = headless
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._tools = BrowserTools()

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

    async def start(self):
        """Start the browser"""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 720}
        )
        self._page = await self._context.new_page()
        self._tools.set_page(self._page)
        logger.info("Browser agent started")

    async def stop(self):
        """Stop the browser"""
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser agent stopped")

    async def navigate(self, url: str) -> BrowserState:
        """Navigate to a URL"""
        result = await self._tools.navigate(url)
        return BrowserState(
            url=result.get("url", ""),
            title=result.get("title", ""),
        )

    async def extract_content(self, selector: Optional[str] = None) -> str:
        """Extract text content from page"""
        result = await self._tools.extract_text(selector)
        return result.get("text", "")

    async def research(
        self,
        query: str,
        max_sources: int = 5,
        max_steps: int = 50,
    ) -> ResearchResult:
        """
        Conduct autonomous research on a topic.

        K2.5 will:
        1. Search for the topic
        2. Navigate to relevant pages
        3. Extract key information
        4. Compile a summary

        Args:
            query: Research query
            max_sources: Maximum sources to visit
            max_steps: Maximum tool call steps

        Returns:
            ResearchResult with summary and sources
        """
        system_prompt = """You are a research agent with browser control capabilities.

Your task is to research a topic by:
1. Navigating to relevant websites (start with search engines or known sources)
2. Extracting key information
3. Following relevant links
4. Compiling findings

Available browser tools:
- browser_navigate: Go to a URL
- browser_click: Click elements
- browser_type: Type in search boxes
- browser_press_key: Press Enter, etc.
- browser_extract_text: Get page content
- browser_get_links: See available links
- browser_screenshot: Capture the page

Be systematic. Start with a search, then explore promising results.
Summarize your findings at the end."""

        tools = BrowserTools.get_tool_schemas()
        tool_map = self._tools.get_tool_map()

        # Execute research task
        sources = []
        screenshots = []
        tool_calls = 0

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research this topic: {query}\n\nFind {max_sources} sources and summarize the key findings."}
        ]

        try:
            for step in range(max_steps):
                response = await self.client.async_client.chat.completions.create(
                    model=self.client.model_id,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.6,
                    max_tokens=4096,
                )

                choice = response.choices[0]

                if choice.finish_reason != "tool_calls":
                    # Research complete
                    return ResearchResult(
                        query=query,
                        summary=choice.message.content or "",
                        sources=sources,
                        screenshots=screenshots,
                        tool_calls_made=tool_calls,
                        success=True,
                    )

                # Process tool calls
                messages.append(choice.message)

                for tool_call in choice.message.tool_calls:
                    import json
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    # Execute tool
                    func = tool_map.get(tool_name)
                    if func:
                        result = await func(**tool_args)
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}

                    # Track sources
                    if tool_name == "browser_navigate" and "url" in result:
                        sources.append({
                            "url": result["url"],
                            "title": result.get("title", ""),
                        })

                    # Track screenshots
                    if tool_name == "browser_screenshot" and "path" in result:
                        screenshots.append(result["path"])

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(result),
                    })

                    tool_calls += 1

            # Max steps reached
            return ResearchResult(
                query=query,
                summary="Research incomplete - max steps reached",
                sources=sources,
                screenshots=screenshots,
                tool_calls_made=tool_calls,
                success=False,
            )

        except Exception as e:
            logger.error(f"Research failed: {e}")
            return ResearchResult(
                query=query,
                summary=f"Research failed: {e}",
                sources=sources,
                screenshots=screenshots,
                tool_calls_made=tool_calls,
                success=False,
            )


if __name__ == "__main__":
    print("Kimi K2.5 Browser Agent")
    print("=" * 50)

    if not PLAYWRIGHT_AVAILABLE:
        print("\nPlaywright not installed!")
        print("Install with: pip install playwright && playwright install")
    else:
        print("\nUsage:")
        print("  async with BrowserAgent() as agent:")
        print("      result = await agent.research('quantum computing')")
        print("      print(result.summary)")
