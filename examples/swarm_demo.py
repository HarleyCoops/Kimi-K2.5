#!/usr/bin/env python3
"""
Kimi K2.5 Swarm Demo - Real MCP Tools

This demo shows how to orchestrate a complex task using:
- Kimi K2.5's swarm orchestrator (up to 100 parallel agents)
- Real MCP tools (ModelScope, web search, etc.)
- Real-time progress monitoring with rich terminal output

Prerequisites:
    1. Set MOONSHOT_API_KEY environment variable
    2. Configure MCP servers in .mcp.json or pass --mcp-config
    3. Install dependencies: pip install -r requirements.txt

Usage:
    # Run with default task
    python examples/swarm_demo.py

    # Run with custom task
    python examples/swarm_demo.py --task "Research quantum computing breakthroughs"

    # Run with specific MCP config
    python examples/swarm_demo.py --mcp-config ~/.config/kimi/mcp.json

    # Verbose mode (show agent reasoning)
    python examples/swarm_demo.py --verbose
"""

import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.layout import Layout
from rich.text import Text
from rich import box

from kimi_client import KimiClient, KimiMode
from swarm.orchestrator import SwarmOrchestrator, SwarmConfig, SwarmResult
from swarm.agents.base_agent import AgentResult, AgentRole
from tools.mcp_bridge import MCPBridge, create_bridge_from_config, load_mcp_config

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

console = Console()


class SwarmMonitor:
    """Real-time swarm execution monitor with rich output"""

    def __init__(self, task: str, verbose: bool = False):
        self.task = task
        self.verbose = verbose
        self.start_time = None
        self.agents_completed = 0
        self.agents_total = 0
        self.tool_calls = 0
        self.current_phase = "Initializing"
        self.agent_results: list[AgentResult] = []
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        )
        self.task_id: TaskID = None

    def on_phase_change(self, phase: str):
        """Called when execution phase changes"""
        self.current_phase = phase
        console.print(f"[cyan]>> {phase}[/cyan]")

    def on_decomposition_complete(self, subtasks: list, parallel_groups: list):
        """Called when task decomposition is complete"""
        self.agents_total = len(subtasks)

        console.print(f"\n[green]Task decomposed into {len(subtasks)} subtasks[/green]")

        # Show subtask table
        table = Table(title="Subtasks", box=box.ROUNDED)
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Role", style="magenta", width=12)
        table.add_column("Task", style="white")
        table.add_column("Group", style="yellow", width=6)

        for i, subtask in enumerate(subtasks):
            # Find which group this subtask is in
            group_num = "?"
            for g_idx, group in enumerate(parallel_groups):
                if subtask.get("id", i) in group or i in group:
                    group_num = str(g_idx + 1)
                    break

            table.add_row(
                str(subtask.get("id", i)),
                subtask.get("role", "custom"),
                subtask.get("task", "")[:60] + "...",
                group_num
            )

        console.print(table)
        console.print(f"\n[dim]Parallel groups: {len(parallel_groups)}[/dim]\n")

    def on_agent_start(self, agent_id: str, role: str, task: str):
        """Called when an agent starts executing"""
        console.print(f"  [yellow]>[/yellow] Starting {role} agent: {agent_id[:30]}")

    def on_agent_complete(self, result: AgentResult):
        """Called when an agent completes"""
        self.agents_completed += 1
        self.tool_calls += result.tool_calls_made
        self.agent_results.append(result)

        status = "[green]OK[/green]" if result.success else "[red]FAILED[/red]"
        console.print(
            f"  [green]<[/green] {result.role.value}: {status} "
            f"(tools: {result.tool_calls_made}, steps: {result.steps_taken})"
        )

        if self.verbose and result.output:
            console.print(f"    [dim]{result.output[:200]}...[/dim]")

    def on_tool_call(self, agent_id: str, tool_name: str, args: dict):
        """Called when a tool is invoked"""
        if self.verbose:
            console.print(f"    [dim]-> {tool_name}({list(args.keys())})[/dim]")

    def print_final_result(self, result: SwarmResult):
        """Print the final swarm result"""
        elapsed = result.execution_time_seconds

        # Summary panel
        summary_text = Text()
        summary_text.append(f"\nTask: ", style="bold")
        summary_text.append(f"{result.task}\n\n")
        summary_text.append(f"Status: ", style="bold")
        summary_text.append(
            "SUCCESS" if result.success else "FAILED",
            style="green bold" if result.success else "red bold"
        )
        summary_text.append(f"\n\nExecution Time: ", style="bold")
        summary_text.append(f"{elapsed:.1f}s\n")
        summary_text.append(f"Agents Used: ", style="bold")
        summary_text.append(f"{result.total_agents}\n")
        summary_text.append(f"Tool Calls: ", style="bold")
        summary_text.append(f"{result.total_tool_calls}\n")

        console.print(Panel(summary_text, title="Swarm Execution Complete", border_style="green"))

        if result.summary:
            console.print(Panel(result.summary, title="Summary", border_style="blue"))

        if result.detailed_output and self.verbose:
            console.print(Panel(
                result.detailed_output[:2000] + "..." if len(result.detailed_output) > 2000 else result.detailed_output,
                title="Detailed Output",
                border_style="dim"
            ))


async def run_swarm_demo(
    task: str,
    mcp_config_path: str = None,
    verbose: bool = False,
    max_agents: int = 10,
):
    """
    Run the swarm demo with a complex task.

    Args:
        task: The task to accomplish
        mcp_config_path: Path to MCP config file
        verbose: Show detailed output
        max_agents: Maximum agents to spawn
    """
    monitor = SwarmMonitor(task, verbose)

    console.print(Panel(
        f"[bold]Kimi K2.5 Agent Swarm Demo[/bold]\n\n"
        f"Task: {task}\n"
        f"Max Agents: {max_agents}\n"
        f"MCP Config: {mcp_config_path or 'default'}",
        title="Swarm Initialization",
        border_style="cyan"
    ))

    # Initialize MCP bridge
    console.print("\n[cyan]>> Connecting to MCP servers...[/cyan]")

    mcp_bridge = None
    tools = []
    tool_map = {}

    try:
        # Try to load MCP config and connect
        config = load_mcp_config(mcp_config_path)
        mcp_servers = config.get("mcpServers", {})

        if mcp_servers:
            mcp_bridge = await create_bridge_from_config(mcp_config_path)

            # List discovered tools
            available_tools = mcp_bridge.list_tools()
            if available_tools:
                console.print(f"[green]Connected! Found {len(available_tools)} tools:[/green]")
                for tool in available_tools[:10]:  # Show first 10
                    console.print(f"  - {tool['server']}/{tool['name']}")
                if len(available_tools) > 10:
                    console.print(f"  ... and {len(available_tools) - 10} more")

                tools = mcp_bridge.get_openai_tools()
                tool_map = mcp_bridge.get_tool_map()
        else:
            console.print("[yellow]No MCP servers configured. Running without tools.[/yellow]")
            console.print("[dim]To add tools, create .mcp.json with your MCP server config.[/dim]")

    except Exception as e:
        console.print(f"[yellow]MCP connection failed: {e}[/yellow]")
        console.print("[dim]Continuing without MCP tools...[/dim]")

    # Initialize Kimi client
    console.print("\n[cyan]>> Initializing Kimi K2.5 client...[/cyan]")

    try:
        client = KimiClient(default_mode=KimiMode.SWARM)
        console.print(f"[green]Connected to {client.model_info.display_name}[/green]")
    except ValueError as e:
        console.print(f"[red]Failed to initialize client: {e}[/red]")
        console.print("[yellow]Make sure MOONSHOT_API_KEY is set[/yellow]")
        return

    # Create swarm orchestrator
    swarm_config = SwarmConfig(
        max_agents=max_agents,
        sub_agent_max_steps=50,
        coordination_timeout=300,
    )

    orchestrator = SwarmOrchestrator(
        client=client,
        config=swarm_config,
        tools=tools,
        tool_map=tool_map,
    )

    # Execute the swarm
    console.print("\n[cyan]>> Executing swarm...[/cyan]\n")

    monitor.start_time = datetime.now()

    result = await orchestrator.execute(
        task=task,
        max_agents=max_agents,
        on_agent_complete=monitor.on_agent_complete,
    )

    # Show decomposition info if available
    if result.metadata.get("decomposition"):
        monitor.on_decomposition_complete(
            result.metadata["decomposition"],
            result.metadata.get("parallel_groups", [])
        )

    # Print final results
    console.print("\n")
    monitor.print_final_result(result)

    # Cleanup MCP connections
    if mcp_bridge:
        await mcp_bridge.disconnect()

    return result


async def interactive_mode():
    """Run in interactive mode, accepting tasks from user"""
    console.print(Panel(
        "[bold]Kimi K2.5 Interactive Swarm Mode[/bold]\n\n"
        "Enter complex tasks and watch the agent swarm work.\n"
        "Type 'quit' or 'exit' to stop.\n"
        "Type 'help' for example tasks.",
        border_style="cyan"
    ))

    example_tasks = [
        "Research the latest developments in nuclear fusion energy",
        "Analyze the competitive landscape of cloud computing providers",
        "Create a comprehensive guide to machine learning frameworks",
        "Investigate climate change mitigation strategies",
        "Compare programming languages for web development in 2025",
    ]

    while True:
        console.print("\n")
        task = console.input("[bold cyan]Enter task>[/bold cyan] ")

        if task.lower() in ("quit", "exit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if task.lower() == "help":
            console.print("\n[bold]Example tasks:[/bold]")
            for i, example in enumerate(example_tasks, 1):
                console.print(f"  {i}. {example}")
            continue

        if task.strip().isdigit():
            idx = int(task.strip()) - 1
            if 0 <= idx < len(example_tasks):
                task = example_tasks[idx]
            else:
                console.print("[red]Invalid example number[/red]")
                continue

        if not task.strip():
            continue

        await run_swarm_demo(task, verbose=True)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Kimi K2.5 Agent Swarm Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run default demo
    python examples/swarm_demo.py

    # Custom task
    python examples/swarm_demo.py --task "Analyze AI trends in 2025"

    # Interactive mode
    python examples/swarm_demo.py --interactive

    # With MCP config
    python examples/swarm_demo.py --mcp-config .mcp.json --verbose
        """
    )

    parser.add_argument(
        "--task", "-t",
        default="Research the current state of large language models, including key players, recent breakthroughs, and future directions",
        help="Task for the swarm to accomplish"
    )
    parser.add_argument(
        "--mcp-config", "-m",
        help="Path to MCP configuration file"
    )
    parser.add_argument(
        "--max-agents", "-a",
        type=int,
        default=10,
        help="Maximum number of agents (default: 10)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including reasoning"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )

    args = parser.parse_args()

    try:
        if args.interactive:
            asyncio.run(interactive_mode())
        else:
            asyncio.run(run_swarm_demo(
                task=args.task,
                mcp_config_path=args.mcp_config,
                verbose=args.verbose,
                max_agents=args.max_agents,
            ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
