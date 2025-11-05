"""
Abstracted client for interacting with various LLMs.
"""

import os
from typing import Dict, Any, List
from loguru import logger
import openai
import anthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential

# NOTE: groq import is lazy in _initialize_client to allow environments without the package to still run
# (we handle ImportError with a clear message).


class LLMClient:
    """A unified client for OpenAI, Anthropic, and Groq LLMs."""

    def __init__(self, config: Dict[str, Any]):
        # If a GROQ_API_KEY exists in environment, prefer/use Groq regardless of config.provider
        env_groq_key = os.getenv("GROQ_API_KEY")
        if env_groq_key:
            self.provider = "groq"
            logger.info(
                "GROQ_API_KEY found in environment — forcing provider to 'groq'."
            )
        else:
            self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 1000)
        self.client = self._initialize_client()
        logger.info(f"Initialized LLM client for provider: {self.provider}")

    def _initialize_client(self):
        """Initializes the appropriate API client based on the provider."""
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            # OpenAI python client (using openai.OpenAI wrapper like in your original code)
            return openai.OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
            return anthropic.Anthropic(api_key=api_key)

        elif self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.")
            try:
                # Lazy import so code doesn't fail on import time if groq isn't installed
                from groq import Groq
            except Exception as e:
                raise ImportError(
                    "The 'groq' package is required for GROQ provider but it's not installed. "
                    "Install it with `pip install groq`.\n"
                    f"Import error: {e}"
                )
            # Groq client supports passing api_key in constructor; base_url can be overridden if needed
            return Groq(api_key=api_key)

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _extract_text_from_response(self, provider: str, response: Any) -> str:
        """
        Tries several common response shapes for chat-style responses and returns text.
        This is tolerant because different SDKs/versions can return slightly different objects.
        """
        # If response is None or falsy
        if not response:
            return ""

        # 1) Try attribute-style access common in many SDKs
        try:
            # openai-python style object: response.choices[0].message.content
            if hasattr(response, "choices"):
                choices = response.choices
                if isinstance(choices, (list, tuple)) and len(choices) > 0:
                    first = choices[0]
                    # nested attribute
                    if hasattr(first, "message") and hasattr(first.message, "content"):
                        return first.message.content
                    # dict-like
                    if isinstance(first, dict):
                        # sometimes message is a dict
                        msg = first.get("message") or first.get("message", {})
                        if isinstance(msg, dict) and "content" in msg:
                            return msg.get("content", "")
                        # sometimes top-level 'text' or 'content' exists
                        if "text" in first:
                            return first.get("text", "")
                        if "content" in first:
                            return first.get("content", "")
        except Exception:
            pass

        # 2) Try dict-like access for typical JSON responses
        try:
            # e.g., response["choices"][0]["message"]["content"]
            if isinstance(response, dict):
                choices = response.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    first = choices[0]
                    if isinstance(first, dict):
                        msg = first.get("message")
                        if isinstance(msg, dict) and "content" in msg:
                            return msg["content"]
                        if "text" in first:
                            return first["text"]
                        if "content" in first:
                            return first["content"]
        except Exception:
            pass

        # 3) Groq SDK sometimes returns response.output_text or similar - try common fallbacks
        try:
            for attr in ("output_text", "text", "content", "message"):
                if hasattr(response, attr):
                    val = getattr(response, attr)
                    if isinstance(val, str):
                        return val
                    # if message is nested object/dict with content
                    if isinstance(val, dict) and "content" in val:
                        return val["content"]
        except Exception:
            pass

        # 4) As a last resort, convert to str
        try:
            return str(response)
        except Exception:
            return ""

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generates a response from the LLM with retries.

        Args:
            prompt: The user prompt.
            system_prompt: An optional system prompt.

        Returns:
            The generated text response.
        """
        try:
            # Build messages in chat format (chronological order)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return self._extract_text_from_response("openai", response)

            elif self.provider == "anthropic":
                # Anthropic's SDKs vary; using the pattern from your original file.
                # If your anthropic client expects a different call shape, adjust accordingly.
                response = self.client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                # Try to parse a few possible shapes; original code returned response.content[0].text
                try:
                    # typical shape in your original: response.content[0].text
                    if hasattr(response, "content"):
                        content = response.content
                        if isinstance(content, (list, tuple)) and len(content) > 0:
                            first = content[0]
                            if hasattr(first, "text"):
                                return first.text
                            if isinstance(first, dict) and "text" in first:
                                return first["text"]
                    # fallback
                    return self._extract_text_from_response("anthropic", response)
                except Exception:
                    return self._extract_text_from_response("anthropic", response)

            elif self.provider == "groq":
                # Groq SDK usage: client.chat.completions.create(messages=[...], model=..., ...)
                # The Groq SDK returns an object similar to other chat SDKs; be tolerant in parsing.
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return self._extract_text_from_response("groq", response)

            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")

        except Exception as e:
            logger.error(f"LLM generation failed for provider={self.provider}: {e}")
            raise

        return ""
