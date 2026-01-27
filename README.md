<div align="center">
  <picture>
      <img src="figures/kimi-logo.png" width="30%" alt="Kimi K2.5: Visual Agentic Intelligence">
  </picture>
</div>

<hr>

<div align="center" style="line-height:1">
  <a href="https://www.kimi.com" target="_blank"><img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-Kimi%20K2.5-ff6b6b?color=1783ff&logoColor=white"/></a>
  <a href="https://www.moonshot.ai" target="_blank"><img alt="Homepage" src="https://img.shields.io/badge/Homepage-Moonshot%20AI-white?logo=Kimi&logoColor=white"/></a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/moonshotai/Kimi-K2.5" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Kimi--K2.5-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://twitter.com/kimi_moonshot" target="_blank"><img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-Kimi.ai-white?logo=x&logoColor=white"/></a>
  <a href="https://discord.gg/TYU2fdJykW" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-Kimi.ai-white?logo=discord&logoColor=white"/></a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/moonshotai/Kimi-K2/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Modified_MIT-f5de53?&color=f5de53"/></a>
</div>

<p align="center">
<b><a href="https://www.kimi.com/blog/kimi-k2-5.html">Tech Blog</a></b> &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; <b><a href="https://huggingface.co/moonshotai/Kimi-K2.5">Model Card</a></b>
</p>

## Install Kimi CLI

```bash
pip install kimi-cli
```

Then start coding with Kimi K2.5 directly in your terminal:

```bash
kimi
```

## Kimi K2.5: Visual Agentic Intelligence

Kimi K2.5 is Moonshot AI's state-of-the-art **native multimodal model** with revolutionary capabilities:

| Feature | Capability |
|---------|------------|
| **Agent Swarm** | Up to 100 sub-agents, 1,500 parallel tool calls, 4.5x speedup |
| **Visual Coding** | Image/video to code, UI debugging, website reconstruction |
| **Thinking Mode** | Deep reasoning with explicit thought traces |
| **Context** | 256K tokens (doubled from K2's 128K) |
| **Multimodal** | Native text + image + video understanding |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your API key
export MOONSHOT_API_KEY="your_api_key"
# Or use our setup utility: python setup_api_key.py

# 3. Run quick start
python examples/quick_start.py
```

### Basic Usage

```python
from kimi_client import KimiClient, KimiMode

# Create client
client = KimiClient()

# Instant mode (fast)
response = client.chat("What is 2+2?", mode=KimiMode.INSTANT)
print(response.content)

# Thinking mode (with reasoning)
response = client.chat("Prove the Pythagorean theorem", mode=KimiMode.THINKING)
print(f"Reasoning: {response.reasoning}")
print(f"Answer: {response.content}")

# Image understanding
response = client.chat_with_image("Convert this UI to code", "screenshot.png")
print(response.content)
```

## Operating Modes

| Mode | Temperature | Best For | Key Feature |
|------|-------------|----------|-------------|
| **INSTANT** | 0.6 | Quick queries, chatbots | Fast, no reasoning traces |
| **THINKING** | 1.0 | Math, coding, research | Visible reasoning process |
| **AGENT** | 0.6 | Autonomous tasks | 200-300 tool steps |
| **SWARM** | 1.0 | Complex workflows | 100 parallel agents |

## Project Structure

```
Kimi-K2/
├── kimi_client.py          # Unified K2.5 client (all modes)
├── config.py               # Model configs and settings
│
├── multimodal/             # Image & video capabilities
│   ├── image_understanding.py
│   └── visual_coding.py    # UI to code generation
│
├── swarm/                  # Multi-agent orchestration
│   ├── orchestrator.py     # Swarm controller
│   └── agents/             # Sub-agent types
│
├── tools/                  # Tool calling infrastructure
│   ├── parallel_executor.py    # 1,500 concurrent calls
│   ├── thinking_tools.py       # 300-step chains
│   └── builtin_tools.py        # Pre-built tools
│
├── live_demos/             # Real-world integrations
│   └── browser_agent.py    # Playwright automation
│
├── examples/               # Quick start guides
│   ├── quick_start.py
│   └── mode_comparison.py
│
└── legacy/                 # K2 code (preserved)
```

## Key Features

### 1. Agent Swarm

Orchestrate up to 100 sub-agents for complex parallel workflows:

```python
from swarm import SwarmOrchestrator

orchestrator = SwarmOrchestrator()
result = await orchestrator.execute(
    "Research the competitive landscape of electric vehicles"
)
print(result.summary)
```

**Capabilities:**
- Automatic task decomposition
- Dynamic agent specialization
- 1,500 parallel tool calls
- 4.5x speedup vs single agent

### 2. Visual Coding

Convert UI designs to production code:

```python
from multimodal import UIToCodeGenerator

generator = UIToCodeGenerator()
result = generator.generate(
    "dashboard.png",
    framework="react",
    typescript=True
)
print(result.code)
```

**Supported Frameworks:**
- HTML/CSS/JavaScript
- React (TypeScript)
- Vue 3
- Tailwind CSS
- Flutter
- React Native

### 3. Extended Tool Chains

Execute 200-300 step tool chains with reasoning:

```python
from tools import ThinkingToolExecutor

executor = ThinkingToolExecutor(log_reasoning=True)
result = executor.execute(
    task="Research and summarize quantum computing papers",
    tools=research_tools,
    tool_map=implementations,
    max_steps=200
)
```

### 4. Browser Automation

AI-controlled web research with Playwright:

```python
from live_demos import BrowserAgent

async with BrowserAgent() as agent:
    result = await agent.research("latest AI developments")
    print(result.summary)
    print(f"Sources: {len(result.sources)}")
```

## Model Architecture

| Specification | Value |
|---------------|-------|
| Architecture | Mixture-of-Experts (MoE) |
| Total Parameters | $1 \times 10^{12}$ (1T) |
| Activated Parameters | $32 \times 10^9$ (32B) |
| Context Length | $256 \times 10^3$ (256K) |
| Vision Encoder | MoonViT (400M params) |
| Attention | Multi-Head Latent Attention |
| Training Data | ~15T mixed visual and text tokens |

## API Configuration

```python
from config import APIConfig, KimiMode, MODE_CONFIGS

# Load from environment
config = APIConfig.from_env()

# Available providers
# - Moonshot (official): https://api.moonshot.ai/v1
# - OpenRouter: https://openrouter.ai/api/v1
# - AIML API: https://api.aimlapi.com/v1
```

**Model IDs:**
- `moonshotai/Kimi-K2.5` - Full multimodal model
- `moonshotai/Kimi-K2-Instruct` - Text-only (legacy)
- `moonshotai/Kimi-K2-Thinking` - Extended reasoning

## Deployment

Supported inference engines:
- **vLLM** - Recommended, tensor/expert parallelism
- **SGLang** - High performance
- **KTransformers** - CPU optimization
- **TensorRT-LLM** - Multi-node inference

See [deploy_guidance.md](docs/deploy_guidance.md) for details.

## Examples

| Example | Description |
|---------|-------------|
| `examples/quick_start.py` | Basic usage for all modes |
| `examples/mode_comparison.py` | Side-by-side mode comparison |
| `multimodal/visual_coding.py` | UI screenshot to code |
| `swarm/orchestrator.py` | Multi-agent research |
| `live_demos/browser_agent.py` | Playwright web automation |

## Migration from K2

This repository has been upgraded from K2 to K2.5. Key changes:

| Feature | K2 | K2.5 |
|---------|-----|------|
| Model | `Kimi-K2-Instruct` | `Kimi-K2.5` |
| Context | 128K | 256K |
| Modality | Text | Text + Image + Video |
| Tool Steps | ~50 | 200-300 |
| Agents | 1 | Up to 100 |

Legacy K2 code is preserved in `legacy/`.

## Resources

- **API**: https://platform.moonshot.ai
- **Chat**: https://www.kimi.com
- **Kimi Code**: https://www.kimi.com/code
- **Tech Blog**: https://www.kimi.com/blog/kimi-k2-5.html
- **Hugging Face**: https://huggingface.co/moonshotai/Kimi-K2.5

## License

Modified MIT License - see [LICENSE](LICENSE)
