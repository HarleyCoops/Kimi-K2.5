#!/usr/bin/env python3
"""
Kimi K2.5 Image Understanding Module

Provides comprehensive image analysis capabilities powered by MoonViT:
- General image analysis and description
- UI/UX screenshot analysis
- Visual debugging (identify errors in UI screenshots)
- Document analysis (charts, diagrams, documents)
- Image Q&A with reasoning traces

Technical Details:
- Vision Encoder: MoonViT (400M parameters)
- Hidden Dimension: 7168
- Native multimodal: Trained on ~15T mixed visual and text tokens
- Context: 256K tokens (can handle multiple images)

Example Usage:
    from multimodal.image_understanding import ImageAnalyzer, analyze_image

    # Quick analysis
    result = analyze_image("screenshot.png", "What's in this image?")

    # Detailed analysis with reasoning
    analyzer = ImageAnalyzer()
    result = analyzer.analyze(
        "dashboard.png",
        analysis_type="ui_review",
        include_reasoning=True
    )
"""

import base64
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass
import logging

from kimi_client import KimiClient, KimiMode, KimiResponse

logger = logging.getLogger(__name__)


@dataclass
class ImageAnalysisResult:
    """Result of image analysis"""
    description: str
    reasoning: Optional[str] = None
    elements: Optional[List[Dict[str, Any]]] = None
    suggestions: Optional[List[str]] = None
    code: Optional[str] = None
    confidence: float = 1.0
    model: str = ""


# Analysis type prompts
ANALYSIS_PROMPTS = {
    "general": (
        "Analyze this image in detail. Describe what you see, including:\n"
        "1. Main subject and composition\n"
        "2. Colors, lighting, and visual style\n"
        "3. Any text or important elements\n"
        "4. Context and potential purpose"
    ),

    "ui_review": (
        "Analyze this UI screenshot as a UX expert. Provide:\n"
        "1. Overall layout and structure description\n"
        "2. Key UI components identified (buttons, forms, navigation, etc.)\n"
        "3. Color scheme and typography assessment\n"
        "4. Usability observations (clarity, accessibility, consistency)\n"
        "5. Specific improvement suggestions\n"
        "Be thorough and specific about element positions and relationships."
    ),

    "visual_debug": (
        "You are a visual debugging expert. Analyze this UI screenshot for issues:\n"
        "1. Layout problems (misalignment, overflow, spacing issues)\n"
        "2. Visual bugs (missing elements, broken images, rendering issues)\n"
        "3. Text issues (truncation, overlap, readability)\n"
        "4. Responsive design problems\n"
        "5. Accessibility concerns (contrast, sizing)\n"
        "List each issue with its location and severity (critical/major/minor)."
    ),

    "extract_text": (
        "Extract all text content from this image. Provide:\n"
        "1. All visible text, preserving structure\n"
        "2. Text hierarchy (headings, body, labels)\n"
        "3. Any text that may be partially obscured\n"
        "Format the output to preserve the original layout as much as possible."
    ),

    "chart_analysis": (
        "Analyze this chart/graph in detail:\n"
        "1. Chart type and title\n"
        "2. Axes labels and scales\n"
        "3. Data series and their values (estimate if not labeled)\n"
        "4. Key trends and patterns\n"
        "5. Insights and conclusions\n"
        "Provide numerical data where visible."
    ),

    "document_analysis": (
        "Analyze this document image:\n"
        "1. Document type and purpose\n"
        "2. Main content summary\n"
        "3. Key information extraction (dates, names, numbers)\n"
        "4. Structure and formatting notes\n"
        "5. Any handwritten annotations if present"
    ),

    "code_screenshot": (
        "Analyze this code screenshot:\n"
        "1. Programming language identification\n"
        "2. Code purpose and functionality\n"
        "3. Transcribe the code accurately\n"
        "4. Identify any visible errors or issues\n"
        "5. Suggest improvements if applicable"
    ),
}


class ImageAnalyzer:
    """
    Comprehensive image analysis using Kimi K2.5 multimodal capabilities.

    Provides various analysis modes optimized for different use cases:
    - General image description
    - UI/UX review and feedback
    - Visual debugging
    - Document and chart analysis
    - Code screenshot transcription
    """

    def __init__(
        self,
        client: Optional[KimiClient] = None,
        default_mode: KimiMode = KimiMode.THINKING,
    ):
        """
        Initialize the ImageAnalyzer.

        Args:
            client: KimiClient instance (creates new one if not provided)
            default_mode: Default analysis mode (THINKING recommended for detailed analysis)
        """
        self.client = client or KimiClient(default_mode=default_mode)
        self.default_mode = default_mode

    def analyze(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        analysis_type: Literal[
            "general", "ui_review", "visual_debug", "extract_text",
            "chart_analysis", "document_analysis", "code_screenshot"
        ] = "general",
        mode: Optional[KimiMode] = None,
        include_reasoning: bool = True,
        **kwargs
    ) -> ImageAnalysisResult:
        """
        Analyze an image with specified analysis type.

        Args:
            image_path: Path to the image file
            prompt: Custom prompt (overrides analysis_type if provided)
            analysis_type: Type of analysis to perform
            mode: KimiMode to use (default: THINKING)
            include_reasoning: Include reasoning trace in result
            **kwargs: Additional parameters for the API call

        Returns:
            ImageAnalysisResult with analysis details
        """
        mode = mode or self.default_mode

        # Build prompt
        if prompt:
            analysis_prompt = prompt
        else:
            analysis_prompt = ANALYSIS_PROMPTS.get(analysis_type, ANALYSIS_PROMPTS["general"])

        # Call API
        response = self.client.chat_with_image(
            message=analysis_prompt,
            image_paths=image_path,
            mode=mode,
            **kwargs
        )

        return ImageAnalysisResult(
            description=response.content,
            reasoning=response.reasoning if include_reasoning else None,
            model=response.model,
        )

    def analyze_multiple(
        self,
        image_paths: List[str],
        prompt: str,
        mode: Optional[KimiMode] = None,
        **kwargs
    ) -> ImageAnalysisResult:
        """
        Analyze multiple images together.

        Args:
            image_paths: List of paths to image files
            prompt: Analysis prompt
            mode: KimiMode to use
            **kwargs: Additional parameters

        Returns:
            ImageAnalysisResult comparing/analyzing all images
        """
        mode = mode or self.default_mode

        response = self.client.chat_with_image(
            message=prompt,
            image_paths=image_paths,
            mode=mode,
            **kwargs
        )

        return ImageAnalysisResult(
            description=response.content,
            reasoning=response.reasoning,
            model=response.model,
        )

    def ui_to_code(
        self,
        image_path: str,
        framework: Literal["html", "react", "vue", "tailwind"] = "html",
        detailed: bool = True,
        **kwargs
    ) -> ImageAnalysisResult:
        """
        Convert UI screenshot to working code.

        Args:
            image_path: Path to UI screenshot
            framework: Target framework for generated code
            detailed: Include detailed styling and responsiveness
            **kwargs: Additional parameters

        Returns:
            ImageAnalysisResult with generated code
        """
        framework_prompts = {
            "html": (
                "Convert this UI design to production-ready HTML, CSS, and JavaScript.\n"
                "Requirements:\n"
                "1. Semantic HTML5 structure\n"
                "2. Modern CSS (flexbox/grid for layout)\n"
                "3. Responsive design with media queries\n"
                "4. Match colors, fonts, and spacing closely\n"
                "5. Include all visible interactive elements\n"
                "6. Add hover states and transitions\n"
                "Provide complete, runnable code."
            ),
            "react": (
                "Convert this UI design to a React component with TypeScript.\n"
                "Requirements:\n"
                "1. Functional component with hooks\n"
                "2. CSS-in-JS or CSS modules for styling\n"
                "3. Proper TypeScript types\n"
                "4. Responsive design\n"
                "5. Semantic HTML structure\n"
                "6. Interactive elements with state management\n"
                "Provide complete, production-ready code."
            ),
            "vue": (
                "Convert this UI design to a Vue 3 component.\n"
                "Requirements:\n"
                "1. Composition API with script setup\n"
                "2. Scoped styling\n"
                "3. TypeScript support\n"
                "4. Responsive design\n"
                "5. All visible interactive elements\n"
                "Provide complete component code."
            ),
            "tailwind": (
                "Convert this UI design to HTML with Tailwind CSS.\n"
                "Requirements:\n"
                "1. Modern Tailwind CSS classes\n"
                "2. Responsive variants (sm:, md:, lg:)\n"
                "3. Match design closely with utility classes\n"
                "4. Semantic HTML structure\n"
                "5. Interactive states (hover:, focus:)\n"
                "Provide complete, runnable HTML with Tailwind."
            ),
        }

        prompt = framework_prompts.get(framework, framework_prompts["html"])

        if detailed:
            prompt += "\n\nBe meticulous about matching:\n- Exact colors (use color picker values)\n- Font sizes and weights\n- Spacing and margins\n- Border radius and shadows\n- Icon placements"

        response = self.client.chat_with_image(
            message=prompt,
            image_paths=image_path,
            mode=KimiMode.THINKING,  # Use thinking for code generation
            max_tokens=16384,  # Allow longer output for complete code
            **kwargs
        )

        return ImageAnalysisResult(
            description="Code generated successfully",
            code=response.content,
            reasoning=response.reasoning,
            model=response.model,
        )

    def visual_debug(
        self,
        image_path: str,
        context: Optional[str] = None,
        **kwargs
    ) -> ImageAnalysisResult:
        """
        Perform visual debugging on a UI screenshot.

        Args:
            image_path: Path to screenshot with potential issues
            context: Additional context about expected behavior
            **kwargs: Additional parameters

        Returns:
            ImageAnalysisResult with identified issues
        """
        prompt = ANALYSIS_PROMPTS["visual_debug"]
        if context:
            prompt += f"\n\nContext: {context}"

        response = self.client.chat_with_image(
            message=prompt,
            image_paths=image_path,
            mode=KimiMode.THINKING,
            **kwargs
        )

        return ImageAnalysisResult(
            description=response.content,
            reasoning=response.reasoning,
            model=response.model,
        )

    def compare_designs(
        self,
        design_path: str,
        implementation_path: str,
        **kwargs
    ) -> ImageAnalysisResult:
        """
        Compare a design mockup with its implementation.

        Args:
            design_path: Path to original design
            implementation_path: Path to implementation screenshot

        Returns:
            ImageAnalysisResult with comparison details
        """
        prompt = (
            "Compare these two images - the first is the original design, "
            "the second is the implementation.\n\n"
            "Provide:\n"
            "1. Overall fidelity score (1-10)\n"
            "2. Specific differences found:\n"
            "   - Color differences\n"
            "   - Spacing/layout differences\n"
            "   - Missing elements\n"
            "   - Typography differences\n"
            "3. Severity of each difference (critical/major/minor)\n"
            "4. Recommendations for achieving pixel-perfect match"
        )

        response = self.client.chat_with_image(
            message=prompt,
            image_paths=[design_path, implementation_path],
            mode=KimiMode.THINKING,
            **kwargs
        )

        return ImageAnalysisResult(
            description=response.content,
            reasoning=response.reasoning,
            model=response.model,
        )


# Convenience functions for quick usage

def analyze_image(
    image_path: str,
    prompt: str = "Describe this image in detail.",
    mode: KimiMode = KimiMode.THINKING,
) -> str:
    """Quick image analysis - returns just the description"""
    analyzer = ImageAnalyzer()
    result = analyzer.analyze(image_path, prompt=prompt, mode=mode)
    return result.description


def image_to_code(
    image_path: str,
    framework: str = "html",
) -> str:
    """Quick UI to code conversion - returns generated code"""
    analyzer = ImageAnalyzer()
    result = analyzer.ui_to_code(image_path, framework=framework)
    return result.code or result.description


def visual_debug(
    image_path: str,
    context: Optional[str] = None,
) -> str:
    """Quick visual debugging - returns identified issues"""
    analyzer = ImageAnalyzer()
    result = analyzer.visual_debug(image_path, context=context)
    return result.description


def describe_ui(image_path: str) -> str:
    """Quick UI description"""
    analyzer = ImageAnalyzer()
    result = analyzer.analyze(image_path, analysis_type="ui_review")
    return result.description


if __name__ == "__main__":
    # Demo usage
    print("Kimi K2.5 Image Understanding Module")
    print("=" * 50)
    print("\nUsage examples:")
    print("  from multimodal.image_understanding import ImageAnalyzer")
    print("  analyzer = ImageAnalyzer()")
    print("  result = analyzer.analyze('screenshot.png', analysis_type='ui_review')")
    print("\nQuick functions:")
    print("  analyze_image('photo.jpg', 'What is this?')")
    print("  image_to_code('ui.png', framework='react')")
    print("  visual_debug('broken_ui.png')")
