#!/usr/bin/env python3
"""
Kimi K2.5 Base Agent

Foundation class for swarm sub-agents. Each agent:
- Has a specialized role and system prompt
- Can execute up to 100 tool steps autonomously
- Reports results back to the orchestrator
- Operates in parallel with other agents

The agent system is inspired by PARL (Parallel-Agent Reinforcement Learning):
- Trainable orchestrator decomposes tasks
- Frozen sub-agents execute subtasks
- No predefined roles or hand-crafted workflows
- Dynamic specialization based on task requirements

Example:
    from swarm.agents import BaseAgent, AgentRole

    # Create custom agent
    agent = BaseAgent(
        role=AgentRole.RESEARCHER,
        task="Find papers on transformer architectures",
        tools=[web_search_tool, read_url_tool]
    )

    result = await agent.execute()
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

from kimi_client import KimiClient, KimiMode, KimiResponse

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Predefined agent roles for common tasks"""

    RESEARCHER = "researcher"
    CODER = "coder"
    ANALYST = "analyst"
    VERIFIER = "verifier"
    WRITER = "writer"
    FACT_CHECKER = "fact_checker"
    SUMMARIZER = "summarizer"
    CRITIC = "critic"
    PLANNER = "planner"
    CUSTOM = "custom"


# Role-specific system prompts
ROLE_PROMPTS = {
    AgentRole.RESEARCHER: """You are a Research Agent in a multi-agent swarm.

Your role:
- Search for relevant information on the given topic
- Gather facts, data, and evidence
- Cite sources accurately
- Identify key findings and insights

Guidelines:
- Be thorough but focused on the task
- Prioritize authoritative sources
- Note any conflicting information
- Summarize findings concisely""",

    AgentRole.CODER: """You are a Coding Agent in a multi-agent swarm.

Your role:
- Write clean, efficient code
- Debug and fix issues
- Follow best practices
- Document your code

Guidelines:
- Use appropriate design patterns
- Handle errors gracefully
- Write testable code
- Optimize for readability first""",

    AgentRole.ANALYST: """You are an Analysis Agent in a multi-agent swarm.

Your role:
- Analyze data and information
- Identify patterns and trends
- Draw insights and conclusions
- Present findings clearly

Guidelines:
- Be data-driven
- Consider multiple perspectives
- Quantify when possible
- Note limitations and uncertainties""",

    AgentRole.VERIFIER: """You are a Verification Agent in a multi-agent swarm.

Your role:
- Validate results from other agents
- Check for accuracy and completeness
- Identify errors or inconsistencies
- Ensure quality standards are met

Guidelines:
- Be thorough and systematic
- Cross-reference multiple sources
- Document any issues found
- Suggest corrections when needed""",

    AgentRole.WRITER: """You are a Writer Agent in a multi-agent swarm.

Your role:
- Create clear, engaging content
- Structure information logically
- Maintain consistent tone and style
- Edit and polish text

Guidelines:
- Adapt style to audience
- Use active voice
- Be concise but complete
- Ensure readability""",

    AgentRole.FACT_CHECKER: """You are a Fact-Checking Agent in a multi-agent swarm.

Your role:
- Verify claims and statements
- Check source credibility
- Identify misinformation
- Provide accurate corrections

Guidelines:
- Use reliable sources
- Be objective
- Document verification process
- Note confidence levels""",

    AgentRole.SUMMARIZER: """You are a Summarizer Agent in a multi-agent swarm.

Your role:
- Condense information effectively
- Capture key points
- Maintain essential meaning
- Create digestible summaries

Guidelines:
- Prioritize important information
- Use clear language
- Maintain accuracy
- Adapt length to requirements""",

    AgentRole.CRITIC: """You are a Critic Agent in a multi-agent swarm.

Your role:
- Evaluate work critically
- Identify weaknesses and gaps
- Suggest improvements
- Provide constructive feedback

Guidelines:
- Be objective and fair
- Support critiques with reasoning
- Balance positives and negatives
- Focus on actionable improvements""",

    AgentRole.PLANNER: """You are a Planner Agent in a multi-agent swarm.

Your role:
- Break down complex tasks
- Create action plans
- Identify dependencies
- Allocate resources efficiently

Guidelines:
- Think systematically
- Consider constraints
- Plan for contingencies
- Prioritize effectively""",

    AgentRole.CUSTOM: """You are a specialized Agent in a multi-agent swarm.

Follow the specific instructions provided for your task.
Coordinate with other agents through the orchestrator.
Report your findings clearly and concisely.""",
}


@dataclass
class AgentConfig:
    """Configuration for an agent"""
    max_steps: int = 100
    temperature: float = 1.0
    max_tokens: int = 4096
    timeout: int = 300  # 5 minutes
    retry_on_error: bool = True
    max_retries: int = 2


@dataclass
class AgentResult:
    """Result from an agent's execution"""
    agent_id: str
    role: AgentRole
    task: str
    output: str
    reasoning: Optional[str] = None
    tool_calls_made: int = 0
    steps_taken: int = 0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """
    Base class for swarm sub-agents.

    Each agent operates semi-autonomously on a specific task,
    using tools as needed to complete the work.
    """

    def __init__(
        self,
        role: AgentRole,
        task: str,
        agent_id: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_map: Optional[Dict[str, Callable]] = None,
        client: Optional[KimiClient] = None,
        config: Optional[AgentConfig] = None,
        custom_prompt: Optional[str] = None,
    ):
        """
        Initialize an agent.

        Args:
            role: Agent's role (determines system prompt)
            task: Specific task for this agent
            agent_id: Unique identifier (auto-generated if not provided)
            tools: Tool definitions for the agent
            tool_map: Mapping of tool names to functions
            client: KimiClient instance
            config: Agent configuration
            custom_prompt: Override system prompt (for CUSTOM role)
        """
        self.role = role
        self.task = task
        self.agent_id = agent_id or f"{role.value}_{id(self)}"
        self.tools = tools or []
        self.tool_map = tool_map or {}
        self.client = client or KimiClient(default_mode=KimiMode.AGENT)
        self.config = config or AgentConfig()

        # Set system prompt
        if custom_prompt:
            self.system_prompt = custom_prompt
        else:
            self.system_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS[AgentRole.CUSTOM])

        # Execution state
        self.messages: List[Dict[str, Any]] = []
        self.tool_calls_count = 0
        self.steps = 0

        logger.info(f"Initialized agent {self.agent_id} with role {role.value}")

    def _build_initial_messages(self) -> List[Dict[str, Any]]:
        """Build initial message list with system prompt and task"""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"TASK: {self.task}\n\nComplete this task using available tools. Report your findings clearly."
            }
        ]

    async def execute(self) -> AgentResult:
        """
        Execute the agent's task.

        Returns:
            AgentResult with output and metadata
        """
        self.messages = self._build_initial_messages()
        self.steps = 0
        self.tool_calls_count = 0
        final_output = ""
        reasoning = None

        try:
            while self.steps < self.config.max_steps:
                # Make API call
                params = {
                    "model": self.client.model_id,
                    "messages": self.messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }

                if self.tools:
                    params["tools"] = self.tools
                    params["tool_choice"] = "auto"

                response = await self.client.async_client.chat.completions.create(**params)
                choice = response.choices[0]

                # Extract reasoning if present
                if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
                    reasoning = choice.message.reasoning_content
                    logger.debug(f"Agent {self.agent_id} reasoning: {reasoning[:100]}...")

                # Check if done
                if choice.finish_reason != "tool_calls":
                    final_output = choice.message.content or ""
                    break

                # Process tool calls
                self.messages.append(choice.message)

                for tool_call in choice.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    logger.debug(f"Agent {self.agent_id} calling tool: {tool_name}")

                    # Execute tool
                    if tool_name in self.tool_map:
                        try:
                            result = await self._execute_tool(tool_name, tool_args)
                        except Exception as e:
                            result = {"error": str(e)}
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}

                    # Append result
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    })

                    self.tool_calls_count += 1

                self.steps += 1

            return AgentResult(
                agent_id=self.agent_id,
                role=self.role,
                task=self.task,
                output=final_output,
                reasoning=reasoning,
                tool_calls_made=self.tool_calls_count,
                steps_taken=self.steps,
                success=True,
            )

        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                role=self.role,
                task=self.task,
                output="",
                error=str(e),
                tool_calls_made=self.tool_calls_count,
                steps_taken=self.steps,
                success=False,
            )

    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Execute a tool, handling both sync and async functions"""
        func = self.tool_map[tool_name]

        if asyncio.iscoroutinefunction(func):
            return await func(**tool_args)
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**tool_args))

    def sync_execute(self) -> AgentResult:
        """Synchronous wrapper for execute()"""
        return asyncio.run(self.execute())


def create_agent(
    role: str,
    task: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_map: Optional[Dict[str, Callable]] = None,
    **kwargs
) -> BaseAgent:
    """
    Factory function to create an agent.

    Args:
        role: Role name (string version of AgentRole)
        task: Task description
        tools: Tool definitions
        tool_map: Tool implementations
        **kwargs: Additional agent configuration

    Returns:
        Configured BaseAgent instance
    """
    try:
        agent_role = AgentRole(role.lower())
    except ValueError:
        agent_role = AgentRole.CUSTOM

    return BaseAgent(
        role=agent_role,
        task=task,
        tools=tools,
        tool_map=tool_map,
        **kwargs
    )


if __name__ == "__main__":
    print("Kimi K2.5 Base Agent Module")
    print("=" * 50)
    print("\nAvailable roles:")
    for role in AgentRole:
        print(f"  - {role.value}")
    print("\nUsage:")
    print("  agent = BaseAgent(role=AgentRole.RESEARCHER, task='Research quantum computing')")
    print("  result = await agent.execute()")
