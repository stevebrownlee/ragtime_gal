"""
Unified template and prompt management module for Ragtime.

This module provides comprehensive template management including:
- Prompt template loading and formatting
- HTML template loading for web UI
- Dynamic context section management
- Style-based template selection
- Legacy format support

Classes:
    PromptManager: Manages prompt templates with dynamic context
    TemplateManager: Manages all templates (prompts + HTML)

Examples:
    >>> from ragtime.utils.templates import TemplateManager
    >>> manager = TemplateManager()
    >>> prompt = manager.get_prompt("standard", "initial", {"context": "...", "question": "..."})
    >>> html = manager.load_html_template()
"""

import os
import json
from typing import Dict, Any, Optional

from ragtime.config.settings import settings
from ragtime.monitoring.logging import get_logger

logger = get_logger(__name__)


class PromptManager:
    """
    Manages prompt templates with dynamic context sections.

    This class handles loading, formatting, and managing prompt templates
    for various query styles and contexts. It supports both legacy and
    modern template formats with dynamic context sections.

    Attributes:
        templates_path: Path to prompt templates JSON file
        templates: Loaded template data
        using_legacy: Whether legacy format is in use

    Examples:
        >>> manager = PromptManager()
        >>> prompt = manager.get_prompt(
        ...     "standard",
        ...     "initial",
        ...     {"context": "Some context", "question": "What is RAG?"}
        ... )
    """

    # Default templates in modern format
    DEFAULT_TEMPLATES = {
        "base_templates": {
            "standard": """
        Answer the question based only on the following context. If the answer is not in the context,
        say "I don't have enough information to answer this question." Don't make up information.

        {context_section}

        Question: {question}

        Answer:
        """,
            "creative": """
        Use your expertise to answer the following question based on the provided context.
        Respond in a creative, engaging style. If the information isn't contained in the context,
        acknowledge this but offer relevant insights where appropriate.

        {context_section}

        Question: {question}

        Answer:
        """
        },
        "system_instructions": {
            "standard": "You are a helpful assistant that provides factual information based on the context provided. If the information isn't in the context, acknowledge this limitation.",
            "creative": "You are a creative assistant that provides engaging and insightful responses. While you prioritize information from the context, you can offer relevant insights beyond it when appropriate."
        },
        "context_formats": {
            "initial": "Context:\n{context}",
            "follow_up": "Previously generated content:\n{previous_content}\n\nAdditional retrieved information:\n{context}",
            "with_previous_content": "{previous_content}\n\nRetrieved Information:\n{context}"
        }
    }

    # Legacy default templates for backward compatibility
    LEGACY_DEFAULT_TEMPLATES = {
        "standard": """
        Answer the question based only on the following context. If the answer is not in the context,
        say "I don't have enough information to answer this question." Don't make up information.

        Context:
        {context}

        Question: {question}

        Answer:
        """,
        "creative": """
        Use your expertise to answer the following question based on the provided context.
        Respond in a creative, engaging style. If the information isn't contained in the context,
        acknowledge this but offer relevant insights where appropriate.

        Context:
        {context}

        Question: {question}

        Answer:
        """
    }

    def __init__(self, templates_path: Optional[str] = None):
        """
        Initialize the prompt manager.

        Args:
            templates_path: Path to templates JSON file (defaults to settings)
        """
        if templates_path is None:
            templates_path = getattr(
                settings,
                'PROMPT_TEMPLATES_PATH',
                './prompt_templates.json'
            )

        self.templates_path = os.path.expanduser(templates_path)
        self.templates = self._load_templates()

        # Check template format
        self.using_legacy = not all(
            key in self.templates
            for key in ['base_templates', 'context_formats']
        )

        if self.using_legacy:
            logger.warning(
                "Using legacy prompt template format",
                extra={"templates_path": self.templates_path}
            )
        else:
            logger.info(
                "Using modern prompt template format",
                extra={"templates_path": self.templates_path}
            )

    def _load_templates(self) -> Dict[str, Any]:
        """
        Load templates from JSON file.

        Returns:
            Dictionary of template data
        """
        try:
            if os.path.exists(self.templates_path):
                with open(self.templates_path, 'r') as f:
                    templates = json.load(f)
                logger.info(
                    "Loaded prompt templates from file",
                    extra={
                        "path": self.templates_path,
                        "num_templates": len(templates)
                    }
                )
                return templates
            else:
                logger.warning(
                    "Prompt templates file not found, using defaults",
                    extra={"path": self.templates_path}
                )
                return self.DEFAULT_TEMPLATES.copy()
        except json.JSONDecodeError as e:
            logger.error(
                "Invalid JSON in templates file",
                extra={"path": self.templates_path, "error": str(e)},
                exc_info=True
            )
            return self.DEFAULT_TEMPLATES.copy()
        except Exception as e:
            logger.error(
                "Error loading prompt templates",
                extra={"path": self.templates_path, "error": str(e)},
                exc_info=True
            )
            return self.DEFAULT_TEMPLATES.copy()

    def get_legacy_template(self, template_name: str) -> Optional[str]:
        """
        Get a template using legacy format.

        Args:
            template_name: Name of the template

        Returns:
            Template string or None if not found
        """
        if self.using_legacy:
            return self.templates.get(template_name)
        else:
            return self.templates.get("legacy_templates", {}).get(template_name)

    def get_template(self, template_name: str = 'standard') -> str:
        """
        Get a specific template by name (legacy compatibility).

        Args:
            template_name: Name of the template (defaults to 'standard')

        Returns:
            Template string
        """
        if template_name in self.templates:
            return self.templates[template_name]
        elif template_name in self.LEGACY_DEFAULT_TEMPLATES:
            logger.warning(
                "Template not found, using default",
                extra={"requested": template_name}
            )
            return self.LEGACY_DEFAULT_TEMPLATES[template_name]
        else:
            logger.warning(
                "Template not found, using standard",
                extra={"requested": template_name}
            )
            return self.LEGACY_DEFAULT_TEMPLATES.get(
                'standard',
                self.DEFAULT_TEMPLATES['base_templates']['standard']
            )

    def get_prompt(
        self,
        style: str,
        query_type: str,
        context_params: Dict[str, str]
    ) -> str:
        """
        Get a formatted prompt using modern template format.

        Args:
            style: Base style (standard, creative, etc.)
            query_type: Query type (initial, follow_up, with_previous_content)
            context_params: Parameters for context formatting

        Returns:
            Formatted prompt string

        Examples:
            >>> manager.get_prompt(
            ...     "standard",
            ...     "initial",
            ...     {"context": "RAG is...", "question": "What is RAG?"}
            ... )
        """
        if self.using_legacy:
            return self._get_legacy_prompt(style, query_type, context_params)

        try:
            # Get base template
            base_template = self.templates["base_templates"].get(style)
            if not base_template:
                logger.warning(
                    "Base template not found, using standard",
                    extra={"requested_style": style}
                )
                base_template = self.templates["base_templates"]["standard"]

            # Get system instruction
            system_instruction = self.templates["system_instructions"].get(
                style, ""
            )

            # Get context format
            context_format = self.templates["context_formats"].get(query_type)
            if not context_format:
                logger.warning(
                    "Context format not found, using initial",
                    extra={"requested_type": query_type}
                )
                context_format = self.templates["context_formats"]["initial"]

            # Format context section
            context_section = context_format.format(**context_params)

            # Combine into final prompt
            prompt = base_template.replace("{context_section}", context_section)

            # Format with remaining parameters
            if "question" in context_params:
                prompt = prompt.format(question=context_params["question"])

            # Add system instruction if present
            if system_instruction:
                prompt = f"{system_instruction}\n\n{prompt}"

            logger.debug(
                "Generated prompt",
                extra={
                    "style": style,
                    "query_type": query_type,
                    "prompt_length": len(prompt)
                }
            )
            return prompt

        except KeyError as e:
            logger.error(
                "Error formatting prompt - missing key",
                extra={"error": str(e), "params": list(context_params.keys())},
                exc_info=True
            )
            # Fall back to legacy or default
            return self._get_fallback_prompt(context_params)
        except Exception as e:
            logger.error(
                "Error formatting prompt",
                extra={"error": str(e)},
                exc_info=True
            )
            return self._get_fallback_prompt(context_params)

    def _get_legacy_prompt(
        self,
        style: str,
        query_type: str,
        context_params: Dict[str, str]
    ) -> str:
        """Get prompt using legacy format."""
        # Map to legacy template name
        if query_type == "follow_up":
            legacy_name = f"follow_up_{style}"
        elif query_type == "with_previous_content":
            legacy_name = f"{style}_with_previous_content"
        else:
            legacy_name = style

        template = self.get_legacy_template(legacy_name)
        if template:
            try:
                return template.format(**context_params)
            except KeyError as e:
                logger.error(
                    "Missing parameter in legacy template",
                    extra={"template": legacy_name, "error": str(e)}
                )
                return self._get_fallback_prompt(context_params)
        else:
            logger.warning(
                "Legacy template not found",
                extra={"name": legacy_name}
            )
            return self._get_fallback_prompt(context_params)

    def _get_fallback_prompt(self, context_params: Dict[str, str]) -> str:
        """Get a basic fallback prompt when formatting fails."""
        context = context_params.get("context", "")
        question = context_params.get("question", "")

        return f"""Answer the question based on the following context.

Context:
{context}

Question: {question}

Answer:"""

    def get_system_instruction(self, style: str) -> str:
        """
        Get system instruction for a given style.

        Args:
            style: Base style (standard, creative, etc.)

        Returns:
            System instruction string or empty string
        """
        if self.using_legacy:
            return ""

        instruction = self.templates.get("system_instructions", {}).get(style, "")
        logger.debug(
            "Retrieved system instruction",
            extra={"style": style, "has_instruction": bool(instruction)}
        )
        return instruction

    def reload_templates(self) -> None:
        """Reload templates from file."""
        logger.info("Reloading prompt templates")
        self.templates = self._load_templates()
        self.using_legacy = not all(
            key in self.templates
            for key in ['base_templates', 'context_formats']
        )


class HTMLTemplateManager:
    """
    Manages HTML template loading for the web UI.

    Attributes:
        template_path: Path to HTML template file

    Examples:
        >>> manager = HTMLTemplateManager()
        >>> html = manager.load_template()
    """

    DEFAULT_ERROR_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #d32f2f; }}
        .error {{ background: #ffebee; padding: 20px; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Template Error</h1>
    <div class="error">
        <p><strong>Could not load template from:</strong> {path}</p>
        <p><strong>Error:</strong> {error}</p>
        <p>Please make sure the template file exists and is readable.</p>
    </div>
</body>
</html>
"""

    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize HTML template manager.

        Args:
            template_path: Path to HTML template file (defaults to settings)
        """
        if template_path is None:
            template_path = getattr(
                settings,
                'TEMPLATE_PATH',
                './template.html'
            )

        self.template_path = os.path.expanduser(template_path)
        logger.info(
            "Initialized HTML template manager",
            extra={"template_path": self.template_path}
        )

    def get_template_path(self) -> str:
        """
        Get the path to the HTML template file.

        Returns:
            Expanded template path
        """
        return self.template_path

    def load_template(self) -> str:
        """
        Load the HTML template from file.

        Returns:
            HTML template content or error template
        """
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            logger.info(
                "Loaded HTML template",
                extra={
                    "path": self.template_path,
                    "size": len(template_content)
                }
            )
            return template_content
        except FileNotFoundError:
            logger.error(
                "HTML template file not found",
                extra={"path": self.template_path}
            )
            return self.DEFAULT_ERROR_TEMPLATE.format(
                path=self.template_path,
                error="File not found"
            )
        except PermissionError:
            logger.error(
                "Permission denied reading HTML template",
                extra={"path": self.template_path}
            )
            return self.DEFAULT_ERROR_TEMPLATE.format(
                path=self.template_path,
                error="Permission denied"
            )
        except Exception as e:
            logger.error(
                "Error loading HTML template",
                extra={"path": self.template_path, "error": str(e)},
                exc_info=True
            )
            return self.DEFAULT_ERROR_TEMPLATE.format(
                path=self.template_path,
                error=str(e)
            )


class TemplateManager:
    """
    Unified template manager for both prompt and HTML templates.

    This is the main interface for all template operations in Ragtime,
    combining prompt template management and HTML template loading.

    Attributes:
        prompt_manager: PromptManager instance
        html_manager: HTMLTemplateManager instance

    Examples:
        >>> manager = TemplateManager()
        >>> prompt = manager.get_prompt("standard", "initial", {...})
        >>> html = manager.load_html_template()
    """

    def __init__(
        self,
        prompt_templates_path: Optional[str] = None,
        html_template_path: Optional[str] = None
    ):
        """
        Initialize the unified template manager.

        Args:
            prompt_templates_path: Path to prompt templates JSON
            html_template_path: Path to HTML template
        """
        self.prompt_manager = PromptManager(prompt_templates_path)
        self.html_manager = HTMLTemplateManager(html_template_path)
        logger.info("Initialized unified TemplateManager")

    # Prompt template methods
    def get_prompt(
        self,
        style: str,
        query_type: str,
        context_params: Dict[str, str]
    ) -> str:
        """
        Get a formatted prompt (delegates to PromptManager).

        Args:
            style: Base style (standard, creative, etc.)
            query_type: Query type (initial, follow_up, etc.)
            context_params: Parameters for context formatting

        Returns:
            Formatted prompt string
        """
        return self.prompt_manager.get_prompt(style, query_type, context_params)

    def get_template(self, template_name: str = 'standard') -> str:
        """
        Get a template by name (legacy compatibility).

        Args:
            template_name: Name of the template

        Returns:
            Template string
        """
        return self.prompt_manager.get_template(template_name)

    def get_system_instruction(self, style: str) -> str:
        """
        Get system instruction for a style.

        Args:
            style: Base style

        Returns:
            System instruction string
        """
        return self.prompt_manager.get_system_instruction(style)

    def reload_prompt_templates(self) -> None:
        """Reload prompt templates from file."""
        self.prompt_manager.reload_templates()

    # HTML template methods
    def load_html_template(self) -> str:
        """
        Load the HTML template.

        Returns:
            HTML template content
        """
        return self.html_manager.load_template()

    def get_html_template_path(self) -> str:
        """
        Get the HTML template path.

        Returns:
            Path to HTML template
        """
        return self.html_manager.get_template_path()


# Convenience functions for backward compatibility
_default_prompt_manager: Optional[PromptManager] = None
_default_html_manager: Optional[HTMLTemplateManager] = None


def get_templates_file() -> str:
    """
    Get the path to the prompt templates file (legacy compatibility).

    Returns:
        Path to templates file
    """
    return os.path.expanduser(
        getattr(settings, 'PROMPT_TEMPLATES_PATH', './prompt_templates.json')
    )


def load_templates() -> Dict[str, Any]:
    """
    Load prompt templates (legacy compatibility).

    Returns:
        Dictionary of templates
    """
    global _default_prompt_manager
    if _default_prompt_manager is None:
        _default_prompt_manager = PromptManager()
    return _default_prompt_manager.templates


def get_template(template_name: str = 'standard') -> str:
    """
    Get a specific prompt template by name (legacy compatibility).

    Args:
        template_name: Name of the template

    Returns:
        Template string
    """
    global _default_prompt_manager
    if _default_prompt_manager is None:
        _default_prompt_manager = PromptManager()
    return _default_prompt_manager.get_template(template_name)


def get_template_path() -> str:
    """
    Get the HTML template path (legacy compatibility).

    Returns:
        Path to HTML template
    """
    return os.path.expanduser(
        getattr(settings, 'TEMPLATE_PATH', './template.html')
    )


def load_html_template() -> str:
    """
    Load the HTML template (legacy compatibility).

    Returns:
        HTML template content
    """
    global _default_html_manager
    if _default_html_manager is None:
        _default_html_manager = HTMLTemplateManager()
    return _default_html_manager.load_template()