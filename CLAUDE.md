# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Kimi K2.5 Overview

This repository implements a comprehensive client library for **Kimi K2.5**, Moonshot AI's native multimodal model with agent swarm capabilities. The codebase has been upgraded from K2 to support K2.5's revolutionary features.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
export MOONSHOT_API_KEY="your_api_key"
# Or use: python setup_api_key.py
```

### Running Examples
```bash
# Quick start
python examples/quick_start.py

# Mode comparison
python examples/mode_comparison.py

# Test configuration
python config.py
```

## Code Architecture

### Core Modules

| Module | Purpose |
|--------|---------|
| `kimi_client.py` | Unified client supporting all K2.5 modes |
| `config.py` | Model configs, API settings, mode parameters |
| `multimodal/` | Image understanding, visual coding |
| `swarm/` | Multi-agent orchestration |
| `tools/` | Tool calling infrastructure |
| `live_demos/` | Browser automation, real integrations |
| `examples/` | Quick start guides |
| `legacy/` | Preserved K2 code |

### Key Classes

- **`KimiClient`**: Main client class with mode switching, streaming, multimodal support
- **`KimiMode`**: Enum for INSTANT, THINKING, AGENT, SWARM modes
- **`SwarmOrchestrator`**: Multi-agent task decomposition and execution
- **`UIToCodeGenerator`**: Screenshot to code conversion
- **`BrowserAgent`**: Playwright-powered web automation

## K2.5 Mode Configuration

| Mode | Temperature | Use Case |
|------|-------------|----------|
| `INSTANT` | 0.6 | Fast responses, chatbots |
| `THINKING` | 1.0 | Complex reasoning, math, coding |
| `AGENT` | 0.6 | Autonomous tool use (200-300 steps) |
| `SWARM` | 1.0 | Multi-agent parallel workflows |

### Mode Selection
```python
from kimi_client import KimiClient, KimiMode

client = KimiClient()

# Fast response
response = client.chat("Quick question", mode=KimiMode.INSTANT)

# With reasoning trace
response = client.chat("Complex problem", mode=KimiMode.THINKING)
print(response.reasoning)  # Access thinking process
```

## API Integration

### Model Names
- **K2.5**: `moonshotai/Kimi-K2.5` (multimodal, recommended)
- **K2 Instruct**: `moonshotai/Kimi-K2-Instruct` (text-only, legacy)
- **K2 Thinking**: `moonshotai/Kimi-K2-Thinking` (extended reasoning)

### Endpoints
- **Official**: `https://api.moonshot.ai/v1`
- **OpenRouter**: `https://openrouter.ai/api/v1`
- **AIML API**: `https://api.aimlapi.com/v1`

### Key Parameters
- Context window: 256K tokens (K2.5), 128K (K2)
- Max completion: 96K tokens for reasoning tasks
- Temperature: 0.6 (instant/agent), 1.0 (thinking/swarm)

## Tool Calling

### Standard Pattern
```python
response = client.execute_with_tools(
    message="Task description",
    tools=tool_definitions,
    tool_map={"tool_name": tool_function},
    max_steps=300  # K2.5 supports up to 300 steps
)
```

### Critical Implementation Notes
- **Unpack arguments**: Use `tool_function(**tool_call_arguments)` not `tool_function(tool_call_arguments)`
- **Append assistant message before tools**: Required for streaming mode
- **Tool call parser**: Use `--tool-call-parser kimi_k2` for vLLM/SGLang

### Parallel Execution
```python
from tools import ParallelToolExecutor

executor = ParallelToolExecutor(max_concurrent=100)
results = await executor.execute_batch(tool_calls, tool_map)
# Supports up to 1,500 concurrent calls
```

## Multimodal Capabilities

### Image Input
```python
response = client.chat_with_image(
    "Describe this UI",
    "screenshot.png",
    mode=KimiMode.THINKING
)
```

### Visual Coding
```python
from multimodal import UIToCodeGenerator

generator = UIToCodeGenerator()
result = generator.generate("ui.png", framework="react")
```

### Supported Formats
- Images: PNG, JPG, JPEG, GIF, WEBP, BMP
- Video: MP4, WEBM, AVI, MOV (official API only)

## Agent Swarm

### Architecture
```
Orchestrator (thinking mode)
    ├── Task decomposition
    ├── Agent creation (up to 100)
    └── Result aggregation

Sub-agents (parallel execution)
    ├── ResearchAgent
    ├── CodingAgent
    ├── AnalysisAgent
    ├── VerificationAgent
    └── WriterAgent
```

### Usage
```python
from swarm import SwarmOrchestrator

orchestrator = SwarmOrchestrator()
result = await orchestrator.execute("Complex research task")
print(result.summary)
print(f"Agents used: {result.total_agents}")
print(f"Tool calls: {result.total_tool_calls}")
```

## Deployment

### Supported Engines
- **vLLM**: Tensor/Expert Parallelism
- **SGLang**: High performance
- **KTransformers**: CPU with AMX
- **TensorRT-LLM**: Multi-node

### Required Flags
```bash
# vLLM
--tool-call-parser kimi_k2
--enable-auto-tool-choice

# Config
"model_type": "kimi_k2"
```

## Testing

```bash
# Test config
python config.py

# Test client
python kimi_client.py

# Test tools
python tools/builtin_tools.py

# Full quick start
python examples/quick_start.py
```

## Mathematical Notation

The repository uses LaTeX/KaTeX in documentation:
- Inline: `$x = y$`
- Display: `$$W \leftarrow W - \eta \nabla W$$`

## Migration Notes (K2 → K2.5)

| Change | K2 | K2.5 |
|--------|-----|------|
| Model | `Kimi-K2-Instruct` | `Kimi-K2.5` |
| Context | 128K | 256K |
| Modality | Text | Text + Image + Video |
| Tool steps | ~50 | 200-300 |
| Agents | 1 | Up to 100 |
| Parallel calls | Sequential | 1,500 concurrent |

Legacy K2 code preserved in `legacy/` directory.
