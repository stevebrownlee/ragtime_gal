"""
prompts.py - Configurable prompt templates for query responses
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

# Default templates
DEFAULT_TEMPLATES = {
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
        """,

    "sixthwood": """
        Use your expertise in the Sixth Wood series to answer the following question based on the provided context.
        Respond in the style and tone consistent with the world of the Sixth Wood.
        If the information isn't contained in the context, acknowledge this but offer relevant insights from your
        knowledge of the series where appropriate.

        Context:
        {context}

        Question: {question}

        Answer:
        """
}

def get_templates_file():
    """Get the path to the templates file, with user home directory expansion"""
    templates_path = os.getenv('PROMPT_TEMPLATES_PATH', './prompt_templates.json')
    return os.path.expanduser(templates_path)

def load_templates():
    """Load templates from the JSON file or use defaults if file doesn't exist"""
    templates_file = get_templates_file()

    if os.path.exists(templates_file):
        try:
            with open(templates_file, 'r') as f:
                templates = json.load(f)
            logger.info(f"Loaded prompt templates from {templates_file}")
            return templates
        except Exception as e:
            logger.error(f"Error loading templates from {templates_file}: {str(e)}")
            logger.info("Using default templates instead")
            return DEFAULT_TEMPLATES
    else:
        # Create the templates file with defaults if it doesn't exist
        try:
            with open(templates_file, 'w') as f:
                json.dump(DEFAULT_TEMPLATES, f, indent=2)
            logger.info(f"Created default templates file at {templates_file}")
        except Exception as e:
            logger.error(f"Could not create templates file: {str(e)}")

        return DEFAULT_TEMPLATES

def get_template(template_name='standard'):
    """Get a specific template by name, falling back to standard if not found"""
    templates = load_templates()

    if template_name in templates:
        return templates[template_name]
    else:
        logger.warning(f"Template '{template_name}' not found, using standard template")
        return templates.get('standard', DEFAULT_TEMPLATES['standard'])