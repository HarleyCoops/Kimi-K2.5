"""
Kimi K2.5 Swarm Agents

Pre-built agent types for common swarm tasks:
- ResearchAgent: Web research, fact-checking, literature review
- CodingAgent: Code generation, debugging, review
- AnalysisAgent: Data analysis, pattern recognition
- VerificationAgent: Result validation, testing
- WriterAgent: Documentation, report generation
"""

from .base_agent import BaseAgent, AgentRole, AgentResult, AgentConfig

__all__ = [
    "BaseAgent",
    "AgentRole",
    "AgentResult",
    "AgentConfig",
]
