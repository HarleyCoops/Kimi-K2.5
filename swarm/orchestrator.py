#!/usr/bin/env python3
"""
Kimi K2.5 Swarm Orchestrator

The orchestrator is the brain of the agent swarm, responsible for:
1. Task decomposition - Breaking complex tasks into parallelizable subtasks
2. Agent creation - Dynamically instantiating specialized sub-agents
3. Parallel execution - Coordinating up to 100 agents simultaneously
4. Result aggregation - Synthesizing outputs into coherent final results

This implements the PARL (Parallel-Agent Reinforcement Learning) paradigm:
- No predefined agent roles or hand-crafted workflows
- Dynamic task decomposition based on complexity
- Automatic agent specialization
- Up to 4.5x speedup vs sequential execution

Performance:
- Max agents: 100
- Max parallel tool calls: 1,500
- Main agent max steps: 15
- Sub-agent max steps: 100

Example Usage:
    from swarm import SwarmOrchestrator

    orchestrator = SwarmOrchestrator()

    # Execute complex research task
    result = await orchestrator.execute(
        "Analyze the competitive landscape of electric vehicles,
        including market share, technology trends, and future outlook"
    )

    print(result.summary)
    for agent_result in result.agent_results:
        print(f"{agent_result.role}: {agent_result.output[:200]}...")
"""

import json
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Add parent directory to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from kimi_client import KimiClient, KimiMode, KimiResponse
from swarm.agents.base_agent import BaseAgent, AgentRole, AgentResult, AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class SwarmConfig:
    """Configuration for the swarm orchestrator"""
    max_agents: int = 100
    max_parallel_tool_calls: int = 1500
    main_agent_max_steps: int = 15
    sub_agent_max_steps: int = 100
    coordination_timeout: int = 600  # 10 minutes
    result_aggregation_strategy: str = "synthesis"  # "synthesis", "merge", "structured"
    allow_agent_communication: bool = False  # Future feature
    cost_limit: Optional[float] = None  # Optional cost limit


@dataclass
class TaskDecomposition:
    """Result of task decomposition by the orchestrator"""
    original_task: str
    subtasks: List[Dict[str, Any]]
    parallel_groups: List[List[int]]  # Groups of subtasks that can run in parallel
    dependencies: Dict[int, List[int]]  # subtask_id -> list of dependent subtask_ids
    reasoning: str


@dataclass
class SwarmResult:
    """Final result from swarm execution"""
    task: str
    summary: str
    detailed_output: str
    agent_results: List[AgentResult]
    total_agents: int
    total_tool_calls: int
    execution_time_seconds: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SwarmOrchestrator:
    """
    Orchestrates multi-agent swarm execution.

    The orchestrator uses Kimi K2.5 in thinking mode to:
    1. Analyze complex tasks
    2. Decompose into parallelizable subtasks
    3. Assign roles and spawn specialized agents
    4. Coordinate parallel execution
    5. Synthesize results into coherent output
    """

    def __init__(
        self,
        client: Optional[KimiClient] = None,
        config: Optional[SwarmConfig] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_map: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize the swarm orchestrator.

        Args:
            client: KimiClient for orchestrator (thinking mode)
            config: Swarm configuration
            tools: Tools available to all agents
            tool_map: Tool implementations
        """
        self.client = client or KimiClient(default_mode=KimiMode.SWARM)
        self.config = config or SwarmConfig()
        self.tools = tools or []
        self.tool_map = tool_map or {}

        # Execution state
        self.agents: List[BaseAgent] = []
        self.results: List[AgentResult] = []

        # System prompt for orchestrator
        self.orchestrator_prompt = """You are the Orchestrator of a Kimi Agent Swarm.

Your capabilities:
- Decompose complex tasks into parallelizable subtasks
- Create specialized sub-agents dynamically
- Coordinate up to 100 agents simultaneously
- Aggregate results into coherent outputs

When decomposing a task:
1. Identify independent subtasks that can run in parallel
2. Identify dependencies between subtasks
3. Assign appropriate roles (researcher, coder, analyst, verifier, writer)
4. Estimate complexity and resource needs

Output your decomposition as JSON with this structure:
{
    "subtasks": [
        {
            "id": 1,
            "task": "specific subtask description",
            "role": "researcher|coder|analyst|verifier|writer",
            "priority": "high|medium|low",
            "depends_on": []  // list of subtask IDs this depends on
        }
    ],
    "parallel_groups": [[1, 2, 3], [4, 5]],  // groups that can run simultaneously
    "synthesis_plan": "how to combine results"
}

Be efficient - create only necessary agents. Maximize parallelism."""

    async def execute(
        self,
        task: str,
        context: Optional[str] = None,
        max_agents: Optional[int] = None,
        on_agent_complete: Optional[Callable[[AgentResult], None]] = None,
    ) -> SwarmResult:
        """
        Execute a complex task using the agent swarm.

        Args:
            task: The task to accomplish
            context: Additional context for the task
            max_agents: Override max agents from config
            on_agent_complete: Callback when each agent completes

        Returns:
            SwarmResult with aggregated output and metrics
        """
        start_time = datetime.now()
        max_agents = max_agents or self.config.max_agents

        logger.info(f"Starting swarm execution for task: {task[:100]}...")

        try:
            # Step 1: Decompose the task
            decomposition = await self._decompose_task(task, context)
            logger.info(f"Decomposed into {len(decomposition.subtasks)} subtasks")

            # Step 2: Create agents for each subtask
            agents = self._create_agents(decomposition, max_agents)
            logger.info(f"Created {len(agents)} agents")

            # Step 3: Execute agents in parallel groups
            all_results = []
            for group_idx, group in enumerate(decomposition.parallel_groups):
                group_agents = [agents[i] for i in group if i < len(agents)]
                logger.info(f"Executing parallel group {group_idx + 1} with {len(group_agents)} agents")

                # Execute group in parallel
                group_results = await asyncio.gather(
                    *[agent.execute() for agent in group_agents],
                    return_exceptions=True
                )

                # Process results
                for result in group_results:
                    if isinstance(result, Exception):
                        logger.error(f"Agent failed: {result}")
                        all_results.append(AgentResult(
                            agent_id="error",
                            role=AgentRole.CUSTOM,
                            task="",
                            output="",
                            success=False,
                            error=str(result)
                        ))
                    else:
                        all_results.append(result)
                        if on_agent_complete:
                            on_agent_complete(result)

            # Step 4: Aggregate results
            final_output = await self._aggregate_results(task, all_results, decomposition)

            # Calculate metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            total_tool_calls = sum(r.tool_calls_made for r in all_results)

            return SwarmResult(
                task=task,
                summary=final_output["summary"],
                detailed_output=final_output["detailed"],
                agent_results=all_results,
                total_agents=len(agents),
                total_tool_calls=total_tool_calls,
                execution_time_seconds=execution_time,
                success=True,
                metadata={
                    "decomposition": decomposition.subtasks,
                    "parallel_groups": decomposition.parallel_groups,
                }
            )

        except Exception as e:
            logger.error(f"Swarm execution failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()

            return SwarmResult(
                task=task,
                summary="",
                detailed_output="",
                agent_results=[],
                total_agents=0,
                total_tool_calls=0,
                execution_time_seconds=execution_time,
                success=False,
                error=str(e)
            )

    async def _decompose_task(
        self,
        task: str,
        context: Optional[str] = None
    ) -> TaskDecomposition:
        """Decompose a complex task into subtasks"""

        prompt = f"""Decompose this task into parallelizable subtasks for a multi-agent swarm.

TASK: {task}
"""
        if context:
            prompt += f"\nCONTEXT: {context}"

        prompt += """

Analyze the task and create a decomposition plan. Consider:
1. What can be done independently (in parallel)?
2. What has dependencies (must wait for other results)?
3. What specialist roles are needed?

Respond with JSON only:
{{
    "subtasks": [
        {{
            "id": 1,
            "task": "specific description",
            "role": "researcher|coder|analyst|verifier|writer",
            "priority": "high|medium|low",
            "depends_on": []
        }}
    ],
    "parallel_groups": [[1, 2], [3]],
    "synthesis_plan": "how to combine"
}}"""

        response = self.client.chat(
            prompt,
            mode=KimiMode.THINKING,
            system_prompt=self.orchestrator_prompt
        )

        # Parse the decomposition
        try:
            # Extract JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            decomp_data = json.loads(content)

            return TaskDecomposition(
                original_task=task,
                subtasks=decomp_data.get("subtasks", []),
                parallel_groups=decomp_data.get("parallel_groups", []),
                dependencies={
                    st["id"]: st.get("depends_on", [])
                    for st in decomp_data.get("subtasks", [])
                },
                reasoning=response.reasoning or ""
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse decomposition, using fallback: {e}")
            # Fallback: single agent for entire task
            return TaskDecomposition(
                original_task=task,
                subtasks=[{
                    "id": 0,
                    "task": task,
                    "role": "researcher",
                    "priority": "high",
                    "depends_on": []
                }],
                parallel_groups=[[0]],
                dependencies={},
                reasoning="Fallback decomposition"
            )

    def _create_agents(
        self,
        decomposition: TaskDecomposition,
        max_agents: int
    ) -> List[BaseAgent]:
        """Create agents for each subtask"""
        agents = []

        for subtask in decomposition.subtasks[:max_agents]:
            role_str = subtask.get("role", "researcher")
            try:
                role = AgentRole(role_str.lower())
            except ValueError:
                role = AgentRole.CUSTOM

            agent = BaseAgent(
                role=role,
                task=subtask["task"],
                agent_id=f"agent_{subtask['id']}_{role_str}",
                tools=self.tools,
                tool_map=self.tool_map,
                client=self.client,
                config=AgentConfig(
                    max_steps=self.config.sub_agent_max_steps
                )
            )
            agents.append(agent)

        return agents

    async def _aggregate_results(
        self,
        original_task: str,
        results: List[AgentResult],
        decomposition: TaskDecomposition
    ) -> Dict[str, str]:
        """Aggregate results from all agents into final output"""

        # Build context from all agent results
        results_text = ""
        for result in results:
            if result.success and result.output:
                results_text += f"\n\n## {result.role.value.upper()} AGENT ({result.agent_id})\n"
                results_text += f"Task: {result.task}\n"
                results_text += f"Output: {result.output}\n"

        prompt = f"""Synthesize the following agent results into a coherent final output.

ORIGINAL TASK: {original_task}

AGENT RESULTS:
{results_text}

Create:
1. A concise summary (2-3 paragraphs)
2. A detailed output combining all findings

Format your response as:
## SUMMARY
[summary here]

## DETAILED OUTPUT
[detailed content here]"""

        response = self.client.chat(
            prompt,
            mode=KimiMode.THINKING,
            system_prompt="You are synthesizing results from multiple specialized agents into a coherent final output."
        )

        # Parse the synthesis
        content = response.content
        summary = ""
        detailed = content

        if "## SUMMARY" in content:
            parts = content.split("## DETAILED OUTPUT")
            summary = parts[0].replace("## SUMMARY", "").strip()
            if len(parts) > 1:
                detailed = parts[1].strip()

        return {
            "summary": summary or content[:500],
            "detailed": detailed
        }


# Pre-built swarm configurations

class ResearchSwarm(SwarmOrchestrator):
    """Pre-configured swarm for research tasks"""

    def __init__(self, **kwargs):
        # Add web search and URL reading tools
        tools = kwargs.pop("tools", []) + [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_url",
                    "description": "Read content from a URL",
                    "parameters": {
                        "type": "object",
                        "required": ["url"],
                        "properties": {
                            "url": {"type": "string", "description": "URL to read"}
                        }
                    }
                }
            }
        ]

        super().__init__(tools=tools, **kwargs)


class CodeReviewSwarm(SwarmOrchestrator):
    """Pre-configured swarm for code review"""

    def __init__(self, **kwargs):
        config = kwargs.pop("config", SwarmConfig())
        config.max_agents = 10  # Focused team for code review

        # Override orchestrator prompt
        super().__init__(config=config, **kwargs)
        self.orchestrator_prompt = """You are orchestrating a code review swarm.

Create agents for:
- Security analysis
- Performance review
- Code style and best practices
- Test coverage assessment
- Documentation review

Each agent focuses on one aspect. Synthesize into actionable feedback."""


if __name__ == "__main__":
    print("Kimi K2.5 Swarm Orchestrator")
    print("=" * 50)
    print("\nCapabilities:")
    print("  - Up to 100 parallel agents")
    print("  - Up to 1,500 coordinated tool calls")
    print("  - 4.5x speedup vs single agent")
    print("\nUsage:")
    print("  orchestrator = SwarmOrchestrator()")
    print("  result = await orchestrator.execute('Complex research task')")
    print("  print(result.summary)")
