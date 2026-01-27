#!/usr/bin/env python3
"""
Kimi K2.5 Mode Comparison

Side-by-side comparison of all four K2.5 modes:
- INSTANT: Fast, no reasoning traces (temp 0.6)
- THINKING: Deep reasoning with traces (temp 1.0)
- AGENT: Extended tool calling (200-300 steps)
- SWARM: Multi-agent orchestration (100 agents)

This demo helps you choose the right mode for your use case.

Usage:
    python examples/mode_comparison.py
"""

import sys
import time
sys.path.insert(0, '.')

from kimi_client import KimiClient, KimiMode
from config import MODE_CONFIGS


def compare_modes_simple():
    """Compare modes on a simple question"""
    print("\n" + "="*70)
    print("COMPARISON: Simple Question")
    print("="*70)

    question = "What is the capital of France?"

    client = KimiClient()

    for mode in [KimiMode.INSTANT, KimiMode.THINKING]:
        config = MODE_CONFIGS[mode]
        print(f"\n[{mode.value.upper()}] (temp={config.temperature})")
        print("-" * 50)

        start = time.time()
        response = client.chat(question, mode=mode)
        elapsed = time.time() - start

        print(f"Answer: {response.content[:200]}")
        if response.reasoning:
            print(f"Reasoning: {response.reasoning[:150]}...")
        print(f"Time: {elapsed:.2f}s | Tokens: {response.total_tokens}")


def compare_modes_complex():
    """Compare modes on a complex reasoning question"""
    print("\n" + "="*70)
    print("COMPARISON: Complex Reasoning")
    print("="*70)

    question = """
    A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball.
    How much does the ball cost? Show your reasoning.
    """

    client = KimiClient()

    for mode in [KimiMode.INSTANT, KimiMode.THINKING]:
        config = MODE_CONFIGS[mode]
        print(f"\n[{mode.value.upper()}] (temp={config.temperature})")
        print("-" * 50)

        start = time.time()
        response = client.chat(question, mode=mode)
        elapsed = time.time() - start

        if response.reasoning:
            print(f"REASONING TRACE:")
            print(response.reasoning[:400])
            print("..." if len(response.reasoning) > 400 else "")
            print()

        print(f"FINAL ANSWER:")
        print(response.content[:300])
        print(f"\nTime: {elapsed:.2f}s | Tokens: {response.total_tokens}")


def mode_selection_guide():
    """Print mode selection guide"""
    print("\n" + "="*70)
    print("MODE SELECTION GUIDE")
    print("="*70)

    guide = """
    INSTANT MODE
    -----------
    Best for: Quick queries, simple tasks, low-latency requirements
    Temperature: 0.6
    Reasoning: Hidden (faster)
    Use when: Speed matters more than explanation

    Examples:
    - FAQ answering
    - Quick translations
    - Simple lookups
    - Chatbot responses

    THINKING MODE
    -------------
    Best for: Complex problems, math, coding, research
    Temperature: 1.0
    Reasoning: Visible (reasoning_content field)
    Use when: Need to see/verify reasoning process

    Examples:
    - Mathematical proofs
    - Code debugging
    - Research analysis
    - Strategic planning

    AGENT MODE
    ----------
    Best for: Autonomous tasks with tool use
    Temperature: 0.6
    Tool Steps: 200-300 consecutive invocations
    Use when: Task requires external actions

    Examples:
    - Web research
    - File operations
    - API interactions
    - Automated workflows

    SWARM MODE
    ----------
    Best for: Complex parallel workflows
    Temperature: 1.0
    Agents: Up to 100 sub-agents
    Tool Calls: Up to 1,500 parallel
    Speedup: 4.5x vs single agent
    Use when: Task is decomposable into parallel subtasks

    Examples:
    - Multi-source research
    - Code review (security, performance, style)
    - Data pipelines
    - Competitive analysis
    """
    print(guide)


def print_config_table():
    """Print configuration table"""
    print("\n" + "="*70)
    print("MODE CONFIGURATIONS")
    print("="*70)

    print(f"\n{'Mode':<12} {'Temp':<8} {'Max Tokens':<12} {'Thinking':<10}")
    print("-" * 50)

    for mode, config in MODE_CONFIGS.items():
        thinking = "Disabled" if config.extra_body and config.extra_body.get("thinking", {}).get("type") == "disabled" else "Enabled"
        print(f"{mode.value:<12} {config.temperature:<8} {config.max_tokens:<12} {thinking:<10}")


def main():
    """Run mode comparison"""
    print("=" * 70)
    print("KIMI K2.5 MODE COMPARISON")
    print("=" * 70)

    try:
        from config import validate_api_key
        if not validate_api_key():
            print("\nNote: API key not set. Showing guide only.")
            mode_selection_guide()
            print_config_table()
            return

        # Run comparisons
        compare_modes_simple()
        compare_modes_complex()
        mode_selection_guide()
        print_config_table()

    except Exception as e:
        print(f"\nError: {e}")
        mode_selection_guide()
        print_config_table()


if __name__ == "__main__":
    main()
