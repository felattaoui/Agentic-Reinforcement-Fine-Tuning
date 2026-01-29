"""
Graders module for RFT training.

This module provides:
- grade(): The main grading function used during RFT training
- GRADER_CODE: The raw Python code string for Azure OpenAI
- get_grader_config(): Returns the grader configuration for training jobs
"""

from src.graders.grader import grade, GRADER_CODE, VALID_TOOLS

__all__ = ["grade", "GRADER_CODE", "VALID_TOOLS", "get_grader_config"]


def get_grader_config() -> dict:
    """
    Get the grader configuration for Azure OpenAI RFT jobs.
    
    Returns:
        dict: Grader configuration with type, name, and source code
    """
    return {
        "type": "python",
        "name": "planner_grader",
        "source": GRADER_CODE
    }
