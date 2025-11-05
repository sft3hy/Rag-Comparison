"""
Central repository for all prompt templates used in the RAG pipelines.
"""


class PromptTemplates:
    """Container for prompt templates."""

    OCR_TEXT_VEC_SYSTEM_PROMPT = """You are an expert AI assistant specializing in data analysis from charts and tables.
Your task is to answer questions based *only* on the provided context, which is text extracted via OCR from a chart.
Be precise and cite the context number you used for your answer. If the information is not in the context, state that clearly."""

    OCR_TEXT_VEC_USER_PROMPT_TEMPLATE = """
Context from OCR:
---
{context_str}
---
Question: {query}
Answer:
"""

    DERENDER_TABLE_VEC_SYSTEM_PROMPT = """You are an expert AI assistant for analyzing structured data.
Answer the question using *only* the data from the provided tables.
Format your answer clearly. If the data is insufficient to answer, say so."""

    DERENDER_TABLE_VEC_USER_PROMPT_TEMPLATE = """
Tables extracted from charts:
---
{context_str}
---
Question: {query}
Answer:
"""

    @staticmethod
    def get_formatted_prompt(template: str, **kwargs) -> str:
        """Formats a prompt template with the given arguments."""
        return template.format(**kwargs)
