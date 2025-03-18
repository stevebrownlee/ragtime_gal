# template.py
# Responsible for loading the HTML template file

import os
import logging

logger = logging.getLogger(__name__)

def get_template_path():
    """Return the path to the HTML template file"""
    # Default is in the same directory as this file
    template_path = os.getenv('TEMPLATE_PATH', './template.html')
    return os.path.expanduser(template_path)

def load_html_template():
    """Load the HTML template from file"""
    template_path = get_template_path()

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
            logger.info("Loaded HTML template from %s", template_path)
            return template_content
    except Exception as e:
        logger.error("Error loading HTML template from %s: %s", template_path, str(e))
        # Return a basic error template if the main template can't be loaded
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Template Error</h1>
            <p>Could not load template from {template_path}: {str(e)}</p>
            <p>Please make sure the template file exists and is readable.</p>
        </body>
        </html>
        """