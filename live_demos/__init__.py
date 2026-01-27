"""
Kimi K2.5 Live Demos

Real-world demonstrations with live integrations:
- BrowserAgent: Playwright-powered web automation
- APIIntegrations: Real API demos (weather, news, etc.)
- DocumentPipeline: End-to-end document processing

These demos showcase K2.5's capabilities with actual services,
not just mock data.

Example:
    from live_demos import BrowserAgent

    agent = BrowserAgent()
    await agent.research("latest AI developments")
"""

from .browser_agent import BrowserAgent, BrowserTools

__all__ = [
    "BrowserAgent",
    "BrowserTools",
]
