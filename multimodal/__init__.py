"""
Kimi K2.5 Multimodal Module

This module provides native multimodal capabilities powered by MoonViT:
- Image understanding and analysis
- UI screenshot to code generation
- Visual debugging
- Video analysis (official API only)
- Document processing

Architecture:
- Vision Encoder: MoonViT (400M parameters)
- Hidden Dimension: 7168
- Native multimodal training: ~15T mixed visual and text tokens
"""

from .image_understanding import (
    ImageAnalyzer,
    analyze_image,
    image_to_code,
    visual_debug,
    describe_ui,
)

from .visual_coding import (
    UIToCodeGenerator,
    screenshot_to_html,
    screenshot_to_react,
    debug_ui_screenshot,
)

__all__ = [
    # Image understanding
    "ImageAnalyzer",
    "analyze_image",
    "image_to_code",
    "visual_debug",
    "describe_ui",
    # Visual coding
    "UIToCodeGenerator",
    "screenshot_to_html",
    "screenshot_to_react",
    "debug_ui_screenshot",
]
