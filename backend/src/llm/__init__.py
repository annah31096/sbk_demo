"""Clients for local language models."""

from backend.src.llm.ollama import LocalModelError, LocalOllamaClient

__all__ = ["LocalModelError", "LocalOllamaClient"]
