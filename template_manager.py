"""
template_manager.py - Enhanced template management for conversational RAG
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TemplateManager:
    """
    Manages prompt templates with dynamic context sections.
    This class replaces the direct template loading in prompts.py with a more
    flexible approach that supports unified templates with dynamic context sections.
    """

    def __init__(self, templates_path: Optional[str] = None):
        """
        Initialize the template manager.

        Args:
            templates_path: Path to the templates JSON file. If None, uses the
                           environment variable PROMPT_TEMPLATES_PATH or defaults
                           to './prompt_templates.json'
        """
        if templates_path is None:
            templates_path = os.getenv('PROMPT_TEMPLATES_PATH', './prompt_templates.json')

        self.templates_path = os.path.expanduser(templates_path)
        self.templates = self._load_templates()

        # Check if we have the new template structure
        self.using_legacy = not all(key in self.templates for key in ['base_templates', 'context_formats'])
        if self.using_legacy:
            logger.warning("Using legacy template format. Consider updating to the new format.")

    def _load_templates(self) -> Dict[str, Any]:
        """Load templates from the JSON file"""
        try:
            if os.path.exists(self.templates_path):
                with open(self.templates_path, 'r') as f:
                    templates = json.load(f)
                logger.info(f"Loaded prompt templates from {self.templates_path}")
                return templates
            else:
                logger.error(f"Templates file not found at {self.templates_path}")
                return self._get_default_templates()
        except Exception as e:
            logger.error(f"Error loading templates from {self.templates_path}: {str(e)}")
            return self._get_default_templates()

    def _get_default_templates(self) -> Dict[str, Any]:
        """Get default templates in the new format"""
        return {
            "base_templates": {
                "standard": "\n        Answer the question based only on the following context. If the answer is not in the context,\n        say \"I don't have enough information to answer this question.\" Don't make up information.\n\n        {context_section}\n\n        Question: {question}\n\n        Answer:\n        ",
                "creative": "\n        Use your expertise to answer the following question based on the provided context.\n        Respond in a creative, engaging style. If the information isn't contained in the context,\n        acknowledge this but offer relevant insights where appropriate.\n\n        {context_section}\n\n        Question: {question}\n\n        Answer:\n        "
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

    def get_legacy_template(self, template_name: str) -> Optional[str]:
        """
        Get a template using the legacy format.

        Args:
            template_name: Name of the template to retrieve

        Returns:
            The template string or None if not found
        """
        if self.using_legacy:
            return self.templates.get(template_name)
        else:
            return self.templates.get("legacy_templates", {}).get(template_name)

    def get_prompt(self, style: str, query_type: str, context_params: Dict[str, str]) -> str:
        """
        Get a prompt using the new template format.

        Args:
            style: The base style (standard, creative, sixthwood)
            query_type: The type of query (initial, follow_up, with_previous_content)
            context_params: Parameters for context formatting

        Returns:
            A formatted prompt string
        """
        if self.using_legacy:
            # Map to legacy template name
            if query_type == "follow_up":
                legacy_name = f"follow_up_{style}"
            elif query_type == "with_previous_content":
                legacy_name = f"{style}_with_previous_content"
            else:
                legacy_name = style

            template = self.get_legacy_template(legacy_name)
            if template:
                # Format with legacy parameters
                return template.format(**context_params)
            else:
                logger.warning(f"Legacy template {legacy_name} not found, using standard")
                return self.get_legacy_template("standard").format(**context_params)

        # Using new template format
        try:
            # Get base template
            base_template = self.templates["base_templates"].get(style)
            if not base_template:
                logger.warning(f"Base template for style '{style}' not found, using standard")
                base_template = self.templates["base_templates"]["standard"]

            # Get system instruction
            system_instruction = self.templates["system_instructions"].get(style, "")

            # Get context format
            context_format = self.templates["context_formats"].get(query_type)
            if not context_format:
                logger.warning(f"Context format for query type '{query_type}' not found, using initial")
                context_format = self.templates["context_formats"]["initial"]

            # Format context section
            context_section = context_format.format(**context_params)

            # Combine into final prompt
            prompt = base_template.replace("{context_section}", context_section)

            # Add system instruction if present
            if system_instruction:
                prompt = f"{system_instruction}\n\n{prompt}"

            return prompt
        except KeyError as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            # Fall back to legacy template as a last resort
            return self.get_legacy_template("standard").format(**context_params)

    def get_system_instruction(self, style: str) -> str:
        """
        Get the system instruction for a given style.

        Args:
            style: The base style (standard, creative, sixthwood)

        Returns:
            The system instruction string or empty string if not found
        """
        if self.using_legacy:
            return ""

        return self.templates["system_instructions"].get(style, "")