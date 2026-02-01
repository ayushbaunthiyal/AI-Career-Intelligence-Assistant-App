"""Prompt templates for the RAG chain."""

from app.prompts.templates import (
    CONDENSE_QUESTION_TEMPLATE,
    QA_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)

__all__ = [
    "SYSTEM_PROMPT",
    "QA_PROMPT_TEMPLATE",
    "CONDENSE_QUESTION_TEMPLATE",
]
