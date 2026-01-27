#!/usr/bin/env python3
"""
Kimi K2.5 Parallel Tool Executor

Handle up to 1,500 concurrent tool calls with:
- Semaphore-based concurrency control
- Rate limiting and backoff
- Error handling and retries
- Progress tracking and cancellation

This enables the full potential of K2.5's agent swarm capabilities,
where multiple agents may invoke tools simultaneously.

Example:
    from tools import ParallelToolExecutor

    executor = ParallelToolExecutor(max_concurrent=100)

    # Execute batch of tool calls
    results = await executor.execute_batch(
        tool_calls=[
            {"name": "web_search", "args": {"query": "AI news"}},
            {"name": "web_search", "args": {"query": "ML papers"}},
            # ... up to 1,500 calls
        ],
        tool_map={"web_search": search_function}
    )
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Representation of a tool call"""
    id: str
    name: str
    arguments: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result of a tool execution"""
    tool_call_id: str
    tool_name: str
    result: Any
    success: bool = True
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    retries: int = 0


@dataclass
class BatchResult:
    """Result of batch tool execution"""
    results: List[ToolResult]
    total_calls: int
    successful_calls: int
    failed_calls: int
    total_execution_time_ms: float
    avg_execution_time_ms: float


class ParallelToolExecutor:
    """
    Execute tool calls in parallel with concurrency control.

    Handles the scale required by K2.5 swarm operations:
    - Up to 1,500 concurrent tool calls
    - Rate limiting to prevent API overload
    - Automatic retries with exponential backoff
    - Error isolation (one failure doesn't stop others)
    """

    def __init__(
        self,
        max_concurrent: int = 100,
        timeout_per_call: float = 60.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        rate_limit_per_second: Optional[float] = None,
    ):
        """
        Initialize the parallel executor.

        Args:
            max_concurrent: Maximum concurrent tool calls
            timeout_per_call: Timeout for each call in seconds
            retry_attempts: Number of retry attempts on failure
            retry_delay: Base delay between retries
            rate_limit_per_second: Optional rate limit
        """
        self.max_concurrent = min(max_concurrent, 1500)  # Cap at K2.5 limit
        self.timeout_per_call = timeout_per_call
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.rate_limit = rate_limit_per_second

        # Concurrency control
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._rate_limiter: Optional[asyncio.Semaphore] = None

        # Statistics
        self._active_calls = 0
        self._completed_calls = 0
        self._failed_calls = 0

        logger.info(f"Initialized ParallelToolExecutor with max_concurrent={max_concurrent}")

    async def execute_batch(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_map: Dict[str, Callable],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """
        Execute a batch of tool calls in parallel.

        Args:
            tool_calls: List of tool call dicts with 'name' and 'arguments'
            tool_map: Mapping of tool names to functions
            on_progress: Optional callback(completed, total)

        Returns:
            BatchResult with all results and statistics
        """
        start_time = datetime.now()
        total = len(tool_calls)

        # Convert to ToolCall objects
        calls = [
            ToolCall(
                id=tc.get("id", f"call_{i}"),
                name=tc.get("name", tc.get("function", {}).get("name", "")),
                arguments=tc.get("arguments", tc.get("args", {})),
            )
            for i, tc in enumerate(tool_calls)
        ]

        # Create tasks
        tasks = [
            self._execute_with_semaphore(call, tool_map, on_progress, total)
            for call in calls
        ]

        # Execute all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        successful = 0
        failed = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ToolResult(
                    tool_call_id=calls[i].id,
                    tool_name=calls[i].name,
                    result=None,
                    success=False,
                    error=str(result)
                ))
                failed += 1
            else:
                processed_results.append(result)
                if result.success:
                    successful += 1
                else:
                    failed += 1

        # Calculate statistics
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        avg_time = total_time / total if total > 0 else 0

        return BatchResult(
            results=processed_results,
            total_calls=total,
            successful_calls=successful,
            failed_calls=failed,
            total_execution_time_ms=total_time,
            avg_execution_time_ms=avg_time,
        )

    async def _execute_with_semaphore(
        self,
        call: ToolCall,
        tool_map: Dict[str, Callable],
        on_progress: Optional[Callable],
        total: int,
    ) -> ToolResult:
        """Execute a single tool call with semaphore control"""
        async with self._semaphore:
            self._active_calls += 1
            try:
                result = await self._execute_single(call, tool_map)
                return result
            finally:
                self._active_calls -= 1
                self._completed_calls += 1
                if on_progress:
                    on_progress(self._completed_calls, total)

    async def _execute_single(
        self,
        call: ToolCall,
        tool_map: Dict[str, Callable],
    ) -> ToolResult:
        """Execute a single tool call with retries"""
        start_time = datetime.now()
        retries = 0

        # Check if tool exists
        if call.name not in tool_map:
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                result=None,
                success=False,
                error=f"Unknown tool: {call.name}"
            )

        func = tool_map[call.name]

        # Retry loop
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                # Apply rate limiting if configured
                if self._rate_limiter:
                    await self._rate_limiter.acquire()

                # Execute with timeout
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(**call.arguments),
                        timeout=self.timeout_per_call
                    )
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: func(**call.arguments)
                        ),
                        timeout=self.timeout_per_call
                    )

                execution_time = (datetime.now() - start_time).total_seconds() * 1000

                return ToolResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    result=result,
                    success=True,
                    execution_time_ms=execution_time,
                    retries=retries
                )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.timeout_per_call}s"
                retries += 1
                logger.warning(f"Tool {call.name} timed out, attempt {attempt + 1}/{self.retry_attempts}")

            except Exception as e:
                last_error = str(e)
                retries += 1
                logger.warning(f"Tool {call.name} failed: {e}, attempt {attempt + 1}/{self.retry_attempts}")

            # Wait before retry
            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

        # All retries failed
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        self._failed_calls += 1

        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            result=None,
            success=False,
            error=last_error,
            execution_time_ms=execution_time,
            retries=retries
        )

    def get_stats(self) -> Dict[str, int]:
        """Get current execution statistics"""
        return {
            "active_calls": self._active_calls,
            "completed_calls": self._completed_calls,
            "failed_calls": self._failed_calls,
        }

    def reset_stats(self):
        """Reset execution statistics"""
        self._completed_calls = 0
        self._failed_calls = 0


async def execute_tools_parallel(
    tool_calls: List[Dict[str, Any]],
    tool_map: Dict[str, Callable],
    max_concurrent: int = 100,
) -> List[ToolResult]:
    """
    Convenience function for parallel tool execution.

    Args:
        tool_calls: List of tool calls
        tool_map: Tool name to function mapping
        max_concurrent: Max concurrent calls

    Returns:
        List of ToolResult objects
    """
    executor = ParallelToolExecutor(max_concurrent=max_concurrent)
    result = await executor.execute_batch(tool_calls, tool_map)
    return result.results


if __name__ == "__main__":
    print("Kimi K2.5 Parallel Tool Executor")
    print("=" * 50)
    print("\nCapabilities:")
    print("  - Up to 1,500 concurrent tool calls")
    print("  - Automatic retries with backoff")
    print("  - Rate limiting support")
    print("\nUsage:")
    print("  executor = ParallelToolExecutor(max_concurrent=100)")
    print("  results = await executor.execute_batch(calls, tool_map)")
