#!/usr/bin/env python3
"""
Swarm Demo with Result Saving
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')

from swarm import SwarmOrchestrator, SwarmConfig
from config import validate_api_key


def save_result(result, output_dir="swarm_results"):
    """Save swarm result to files"""
    
    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"swarm_{timestamp}"
    
    # Save summary as text
    summary_file = out_path / f"{base_name}_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Task: {result.task}\n")
        f.write(f"Success: {result.success}\n")
        f.write(f"Agents: {result.total_agents}\n")
        f.write(f"Tool Calls: {result.total_tool_calls}\n")
        f.write(f"Time: {result.execution_time_seconds:.2f}s\n")
        f.write("=" * 60 + "\n\n")
        f.write(result.summary)
    
    # Save detailed output
    detailed_file = out_path / f"{base_name}_detailed.txt"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        f.write(result.detailed_output)
    
    # Save full result as JSON
    json_file = out_path / f"{base_name}.json"
    result_dict = {
        "task": result.task,
        "success": result.success,
        "summary": result.summary,
        "detailed_output": result.detailed_output,
        "total_agents": result.total_agents,
        "total_tool_calls": result.total_tool_calls,
        "execution_time_seconds": result.execution_time_seconds,
        "error": result.error,
        "agent_results": [
            {
                "agent_id": r.agent_id,
                "role": r.role.value,
                "task": r.task,
                "output": r.output,
                "success": r.success,
                "error": r.error,
                "tool_calls_made": r.tool_calls_made,
                "steps_taken": r.steps_taken,
            }
            for r in result.agent_results
        ]
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to:")
    print(f"  Summary: {summary_file}")
    print(f"  Detailed: {detailed_file}")
    print(f"  JSON: {json_file}")
    
    return summary_file, detailed_file, json_file


async def main():
    if not validate_api_key():
        print("ERROR: MOONSHOT_API_KEY not set!")
        return

    print("=" * 60)
    print("SWARM DEMO WITH SAVE")
    print("=" * 60)

    # Limit agents for faster execution
    config = SwarmConfig(max_agents=3, sub_agent_max_steps=5)
    orchestrator = SwarmOrchestrator(config=config)

    task = "What are the key features of Python 3.12?"
    print(f"\nTask: {task}\n")

    result = await orchestrator.execute(task=task)

    if result.success:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(result.summary[:500] + "..." if len(result.summary) > 500 else result.summary)
        
        # Save results
        save_result(result)
    else:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
