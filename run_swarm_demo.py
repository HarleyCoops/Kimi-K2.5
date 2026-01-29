#!/usr/bin/env python3
"""
Swarm Orchestrator Demo

Run a complex task using multiple parallel agents.
Make sure MOONSHOT_API_KEY is set in your environment.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from swarm import SwarmOrchestrator
from config import validate_api_key


async def main():
    # Check API key
    if not validate_api_key():
        print("ERROR: MOONSHOT_API_KEY not set!")
        print("Set it with: export MOONSHOT_API_KEY='your_key'")
        return

    print("=" * 60)
    print("KIMI K2.5 SWARM ORCHESTRATOR DEMO")
    print("=" * 60)

    # Create the orchestrator
    orchestrator = SwarmOrchestrator()

    # Define a complex task
    task = (
        "Research and analyze: What are the main types of renewable energy? "
        "For each type (solar, wind, hydroelectric, geothermal, biomass), "
        "explain how it works, its current global capacity, advantages, "
        "and main challenges. Provide a summary comparison."
    )

    print(f"\nTask: {task}\n")
    print("Decomposing task and spawning agents...")
    print("-" * 60)

    # Progress callback
    def on_agent_complete(agent_result):
        status = "OK" if agent_result.success else "FAIL"
        print(f"  [{status}] {agent_result.role.value} agent completed")

    # Execute the task
    result = await orchestrator.execute(
        task=task,
        on_agent_complete=on_agent_complete
    )

    print("-" * 60)

    # Display results
    if result.success:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(result.summary)

        print("\n" + "=" * 60)
        print("DETAILED OUTPUT")
        print("=" * 60)
        print(result.detailed_output)

        print("\n" + "=" * 60)
        print("METRICS")
        print("=" * 60)
        print(f"Total agents used: {result.total_agents}")
        print(f"Total tool calls: {result.total_tool_calls}")
        print(f"Execution time: {result.execution_time_seconds:.2f} seconds")
    else:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
