#!/usr/bin/env python3
"""
Kimi K2.5 Visual Coding Module

Convert visual designs to production-ready code with Kimi K2.5's
state-of-the-art visual coding capabilities.

Features:
- UI screenshot to HTML/CSS/JS
- Screenshot to React/Vue/Svelte components
- Figma-style design to code
- Mobile app UI to Flutter/React Native
- Video workflow to code (step extraction)

Kimi K2.5 Visual Coding Benchmark Performance:
- Meaningfully improved over K2 across all task types
- Handles complex layouts with animations
- Produces responsive, accessible code
- Supports interactive element generation

Example Usage:
    from multimodal.visual_coding import UIToCodeGenerator

    generator = UIToCodeGenerator()

    # Convert screenshot to HTML
    code = generator.generate(
        "dashboard.png",
        framework="html",
        responsive=True,
        interactive=True
    )

    # Generate React component
    code = generator.generate(
        "component.png",
        framework="react",
        typescript=True
    )
"""

import base64
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
import logging

from kimi_client import KimiClient, KimiMode, KimiResponse

logger = logging.getLogger(__name__)


class Framework(Enum):
    """Supported output frameworks"""
    HTML = "html"
    REACT = "react"
    VUE = "vue"
    SVELTE = "svelte"
    TAILWIND = "tailwind"
    FLUTTER = "flutter"
    REACT_NATIVE = "react_native"


@dataclass
class CodeGenerationResult:
    """Result of code generation from visual input"""
    code: str
    framework: str
    files: Dict[str, str] = field(default_factory=dict)  # filename -> content
    reasoning: Optional[str] = None
    design_notes: Optional[str] = None
    tokens_used: int = 0


# Framework-specific generation prompts
FRAMEWORK_PROMPTS = {
    Framework.HTML: """
Convert this UI design to production-ready HTML, CSS, and JavaScript.

REQUIREMENTS:
1. Structure:
   - Semantic HTML5 elements (header, nav, main, section, article, footer)
   - Proper heading hierarchy (h1-h6)
   - Accessible form labels and ARIA attributes

2. Styling:
   - Modern CSS (CSS Grid, Flexbox)
   - CSS custom properties for colors and spacing
   - Responsive breakpoints (mobile-first approach)
   - Smooth transitions and hover states

3. Interactivity:
   - Vanilla JavaScript for interactions
   - Event delegation where appropriate
   - Form validation if forms are present

4. Quality:
   - Match colors precisely (extract from image)
   - Accurate spacing and typography
   - Cross-browser compatible
   - Performance optimized (no render-blocking)

OUTPUT FORMAT:
Provide complete files that can be run directly:
- index.html (complete HTML document)
- styles.css (all styles)
- script.js (all JavaScript)
""",

    Framework.REACT: """
Convert this UI design to a React component with TypeScript.

REQUIREMENTS:
1. Component Structure:
   - Functional component with hooks
   - Proper TypeScript interfaces/types
   - Component composition (break into smaller components if complex)

2. Styling:
   - CSS Modules or styled-components
   - Responsive design
   - Theme-aware (support dark mode if applicable)

3. State & Logic:
   - useState for local state
   - useEffect for side effects
   - Custom hooks if logic is reusable

4. Best Practices:
   - Semantic HTML within JSX
   - Accessibility (aria-* attributes)
   - Proper key props for lists
   - Memoization where beneficial

OUTPUT FORMAT:
```tsx
// Component.tsx
```
```css
// Component.module.css (or styled-components)
```
""",

    Framework.VUE: """
Convert this UI design to a Vue 3 component.

REQUIREMENTS:
1. Component Structure:
   - Composition API with <script setup>
   - TypeScript support
   - Proper props and emits definitions

2. Styling:
   - Scoped CSS
   - CSS variables for theming
   - Responsive design

3. Reactivity:
   - ref() and reactive() appropriately
   - computed properties
   - watchers if needed

OUTPUT FORMAT:
```vue
<template>
  <!-- ... -->
</template>

<script setup lang="ts">
// ...
</script>

<style scoped>
/* ... */
</style>
```
""",

    Framework.TAILWIND: """
Convert this UI design to HTML with Tailwind CSS.

REQUIREMENTS:
1. Layout:
   - Tailwind's flexbox/grid utilities
   - Responsive prefixes (sm:, md:, lg:, xl:)
   - Container and spacing utilities

2. Styling:
   - Accurate color matching (use closest Tailwind colors or custom)
   - Typography utilities
   - Shadow and border utilities

3. Interactivity:
   - hover:, focus:, active: states
   - transition utilities
   - group and peer utilities for complex interactions

4. Accessibility:
   - sr-only for screen reader content
   - focus-visible for keyboard navigation

OUTPUT FORMAT:
Complete HTML file with Tailwind CDN or config for custom colors.
""",

    Framework.FLUTTER: """
Convert this UI design to Flutter/Dart code.

REQUIREMENTS:
1. Widget Structure:
   - Proper widget tree composition
   - StatelessWidget or StatefulWidget as appropriate
   - Extract reusable widgets

2. Layout:
   - Column, Row, Stack, Expanded, Flexible
   - Padding, SizedBox for spacing
   - MediaQuery for responsiveness

3. Styling:
   - ThemeData integration
   - Consistent color scheme
   - Custom fonts if visible

OUTPUT FORMAT:
```dart
// main.dart or component file
```
""",

    Framework.REACT_NATIVE: """
Convert this UI design to React Native code.

REQUIREMENTS:
1. Components:
   - React Native core components (View, Text, Image, etc.)
   - Proper StyleSheet usage
   - Platform-specific adjustments if needed

2. Layout:
   - Flexbox for layouts
   - Dimensions API for responsive sizing
   - SafeAreaView for notch/status bar

3. Styling:
   - StyleSheet.create for performance
   - Consistent spacing with constants
   - Dynamic theming support

OUTPUT FORMAT:
```tsx
// Screen/Component file
```
""",
}


class UIToCodeGenerator:
    """
    Generate production-ready code from UI screenshots.

    Uses Kimi K2.5's visual coding capabilities to produce
    accurate, responsive, and interactive code.
    """

    def __init__(
        self,
        client: Optional[KimiClient] = None,
    ):
        """
        Initialize the code generator.

        Args:
            client: KimiClient instance (creates one if not provided)
        """
        self.client = client or KimiClient(default_mode=KimiMode.THINKING)

    def generate(
        self,
        image_path: str,
        framework: Literal["html", "react", "vue", "svelte", "tailwind", "flutter", "react_native"] = "html",
        responsive: bool = True,
        interactive: bool = True,
        typescript: bool = True,
        include_comments: bool = True,
        custom_instructions: Optional[str] = None,
        max_tokens: int = 16384,
        **kwargs
    ) -> CodeGenerationResult:
        """
        Generate code from a UI screenshot.

        Args:
            image_path: Path to UI screenshot
            framework: Target framework
            responsive: Include responsive design
            interactive: Include interactive elements
            typescript: Use TypeScript (for React/Vue)
            include_comments: Add code comments
            custom_instructions: Additional instructions
            max_tokens: Maximum output tokens
            **kwargs: Additional API parameters

        Returns:
            CodeGenerationResult with generated code
        """
        # Get framework prompt
        fw = Framework(framework)
        base_prompt = FRAMEWORK_PROMPTS.get(fw, FRAMEWORK_PROMPTS[Framework.HTML])

        # Build full prompt
        prompt = f"TASK: Convert this UI screenshot to code.\n\n{base_prompt}"

        # Add modifiers
        modifiers = []
        if responsive:
            modifiers.append("- Ensure fully responsive design for all screen sizes")
        if interactive:
            modifiers.append("- Include all interactive states and animations visible in the design")
        if typescript and framework in ["react", "vue", "svelte"]:
            modifiers.append("- Use TypeScript with proper type definitions")
        if include_comments:
            modifiers.append("- Add helpful code comments explaining key sections")
        if not interactive:
            modifiers.append("- Static layout only, no JavaScript/interactivity needed")

        if modifiers:
            prompt += "\n\nADDITIONAL REQUIREMENTS:\n" + "\n".join(modifiers)

        if custom_instructions:
            prompt += f"\n\nCUSTOM INSTRUCTIONS:\n{custom_instructions}"

        prompt += "\n\nAnalyze the design carefully before coding. Pay close attention to spacing, colors, and alignment."

        # Generate code
        response = self.client.chat_with_image(
            message=prompt,
            image_paths=image_path,
            mode=KimiMode.THINKING,
            max_tokens=max_tokens,
            **kwargs
        )

        return CodeGenerationResult(
            code=response.content,
            framework=framework,
            reasoning=response.reasoning,
            tokens_used=response.total_tokens,
        )

    def generate_component_library(
        self,
        image_paths: List[str],
        framework: str = "react",
        library_name: str = "components",
        **kwargs
    ) -> Dict[str, CodeGenerationResult]:
        """
        Generate a component library from multiple UI screenshots.

        Args:
            image_paths: List of component screenshot paths
            framework: Target framework
            library_name: Name for the component library
            **kwargs: Additional parameters

        Returns:
            Dict mapping component names to their CodeGenerationResult
        """
        results = {}

        for i, image_path in enumerate(image_paths):
            component_name = Path(image_path).stem or f"Component{i+1}"

            custom = f"Name this component '{component_name}'. It will be part of a '{library_name}' component library. Ensure consistent styling with other components."

            result = self.generate(
                image_path,
                framework=framework,
                custom_instructions=custom,
                **kwargs
            )

            results[component_name] = result
            logger.info(f"Generated component: {component_name}")

        return results

    def iterate_design(
        self,
        original_image: str,
        feedback: str,
        previous_code: str,
        framework: str = "html",
        **kwargs
    ) -> CodeGenerationResult:
        """
        Iterate on a design based on feedback.

        Args:
            original_image: Path to original design
            feedback: Feedback/requested changes
            previous_code: Previous code iteration
            framework: Target framework

        Returns:
            Updated CodeGenerationResult
        """
        prompt = f"""
TASK: Update the code based on feedback.

ORIGINAL CODE:
```
{previous_code[:5000]}...
```

FEEDBACK/REQUESTED CHANGES:
{feedback}

Please update the code to address the feedback while maintaining fidelity to the original design shown in the image. Only change what's necessary to address the feedback.
"""

        response = self.client.chat_with_image(
            message=prompt,
            image_paths=original_image,
            mode=KimiMode.THINKING,
            **kwargs
        )

        return CodeGenerationResult(
            code=response.content,
            framework=framework,
            reasoning=response.reasoning,
        )


# Convenience functions

def screenshot_to_html(
    image_path: str,
    responsive: bool = True,
    interactive: bool = True,
) -> str:
    """Quick conversion to HTML/CSS/JS"""
    generator = UIToCodeGenerator()
    result = generator.generate(
        image_path,
        framework="html",
        responsive=responsive,
        interactive=interactive,
    )
    return result.code


def screenshot_to_react(
    image_path: str,
    typescript: bool = True,
) -> str:
    """Quick conversion to React component"""
    generator = UIToCodeGenerator()
    result = generator.generate(
        image_path,
        framework="react",
        typescript=typescript,
    )
    return result.code


def debug_ui_screenshot(
    image_path: str,
    expected_code: Optional[str] = None,
) -> str:
    """
    Debug a UI screenshot - identify visual issues and suggest fixes.

    Args:
        image_path: Path to screenshot with potential issues
        expected_code: Optional code that should produce this UI

    Returns:
        Analysis of issues and suggested fixes
    """
    client = KimiClient()

    prompt = """
Analyze this UI screenshot for visual bugs and issues.

IDENTIFY:
1. Layout Problems:
   - Misaligned elements
   - Overflow/clipping issues
   - Incorrect spacing

2. Visual Bugs:
   - Missing or broken images
   - Incorrect colors
   - Font rendering issues

3. Responsive Issues:
   - Elements not fitting container
   - Hidden content
   - Overlapping elements

4. Accessibility Issues:
   - Contrast problems
   - Too-small text
   - Missing focus indicators

For each issue found:
- Describe the problem precisely
- Locate it in the screenshot
- Suggest the CSS/code fix
- Rate severity (critical/major/minor)
"""

    if expected_code:
        prompt += f"\n\nEXPECTED CODE:\n```\n{expected_code[:3000]}\n```\n\nCompare the visual output with this code to identify discrepancies."

    response = client.chat_with_image(
        message=prompt,
        image_paths=image_path,
        mode=KimiMode.THINKING,
    )

    return response.content


if __name__ == "__main__":
    print("Kimi K2.5 Visual Coding Module")
    print("=" * 50)
    print("\nSupported frameworks:")
    for fw in Framework:
        print(f"  - {fw.value}")
    print("\nUsage:")
    print("  from multimodal.visual_coding import UIToCodeGenerator")
    print("  generator = UIToCodeGenerator()")
    print("  result = generator.generate('screenshot.png', framework='react')")
    print("  print(result.code)")
