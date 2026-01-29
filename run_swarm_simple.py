#!/usr/bin/env python3
"""
Simple Swarm Demo - 3 agents
"""

import asyncio
import sys
sys.path.insert(0, '.')

from swarm import SwarmOrchestrator
from config import validate_api_key


async def main():
    if not validate_api_key():
        print("ERROR: MOONSHOT_API_KEY not set!")
        return

    print("=" * 60)
    print("SIMPLE SWARM DEMO (3 Agents)")
    print("=" * 60)

    orchestrator = SwarmOrchestrator()

    # Simpler task that should spawn fewer agents
    task = "Compare Python, JavaScript, and Go for backend web development."

    print(f"\nTask: {task}\n")

    def on_agent_complete(agent_result):
        status = "OK" if agent_result.success else "FAIL"
        print(f"  [{status}] {agent_result.role.value} agent done")

    result = await orchestrator.execute(
        task=task,
        on_agent_complete=on_agent_complete
    )

    if result.success:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(result.summary)
        print(f"\nAgents: {result.total_agents} | Time: {result.execution_time_seconds:.1f}s")
    else:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
