"""
Kimi K2.5 Agent Swarm Module

Revolutionary multi-agent orchestration powered by PARL
(Parallel-Agent Reinforcement Learning).

Capabilities:
- Up to 100 sub-agents running in parallel
- Up to 1,500 coordinated tool calls
- 4.5x speedup vs single-agent execution
- Dynamic agent creation without predefined roles
- Self-directed task decomposition

Architecture:
    ┌─────────────────────────────────────────────┐
    │            SWARM ORCHESTRATOR               │
    │  - Task decomposition                       │
    │  - Sub-agent creation                       │
    │  - Result aggregation                       │
    └────────────────┬────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
    │Research │ │ Coding  │ │ Verify  │  × up to 100
    │ Agent   │ │  Agent  │ │  Agent  │
    └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │
    ┌────▼───────────▼───────────▼────┐
    │    PARALLEL TOOL EXECUTION      │
    │      (up to 1,500 calls)        │
    └─────────────────────────────────┘

Example Usage:
    from swarm import SwarmOrchestrator, ResearchSwarm

    # Create orchestrator
    orchestrator = SwarmOrchestrator()

    # Execute complex task with auto-scaling agents
    result = await orchestrator.execute(
        "Research the latest developments in quantum computing,
        analyze key papers, and produce a comprehensive report"
    )

    # Use pre-built research swarm
    swarm = ResearchSwarm()
    report = await swarm.research("Impact of AI on healthcare")
"""

from .orchestrator import (
    SwarmOrchestrator,
    SwarmResult,
    SwarmConfig,
)

from .agents.base_agent import (
    BaseAgent,
    AgentRole,
    AgentResult,
)

__all__ = [
    "SwarmOrchestrator",
    "SwarmResult",
    "SwarmConfig",
    "BaseAgent",
    "AgentRole",
    "AgentResult",
]
