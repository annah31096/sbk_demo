"""Async client for locally hosted Ollama chat and embedding models."""

import inspect
import json
import logging
from types import SimpleNamespace
from typing import Any, Callable
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


class LocalModelError(RuntimeError):
    """Raised when the local Ollama runtime cannot serve a model request."""


class LocalOllamaClient:
    def __init__(
        self,
        host: str,
        model: str,
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._client = client or httpx.AsyncClient(
            base_url=host.rstrip("/"),
            timeout=timeout,
        )

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: Any | None = None,
        think: bool = False,
        stream: bool = False,
        on_token: Callable[[str], Any] | None = None,
        **_: Any,
    ) -> Any:
        options = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": self._normalize_messages(messages),
            "stream": stream,
            "think": think,
            "options": options,
        }
        if tools:
            payload["tools"] = tools
        if response_format is not None:
            payload["format"] = (
                response_format
                if isinstance(response_format, dict)
                else "json"
            )

        response = (
            await self._stream_chat(payload, on_token)
            if stream
            else await self._post("/api/chat", payload)
        )
        raw_message = response.get("message")
        if not isinstance(raw_message, dict):
            raise LocalModelError("Das lokale Ollama-Modell lieferte keine Antwort.")
        content = raw_message.get("content", "")
        if not isinstance(content, str):
            raise LocalModelError(
                "Das lokale Ollama-Modell lieferte ein ungültiges Antwortformat."
            )

        tool_calls = []
        for call in raw_message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function", {})
            if not isinstance(function, dict):
                continue
            tool_calls.append(
                SimpleNamespace(
                    id=f"ollama-{uuid4().hex}",
                    function=SimpleNamespace(
                        name=function.get("name", ""),
                        arguments=function.get("arguments", {}),
                    ),
                )
            )

        message = SimpleNamespace(
            content=content,
            thinking=raw_message.get("thinking", ""),
            tool_calls=tool_calls,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            done_reason=response.get("done_reason"),
        )

    async def _stream_chat(
        self,
        payload: dict[str, Any],
        on_token: Callable[[str], Any] | None,
    ) -> dict[str, Any]:
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        done_reason = None
        try:
            async with self._client.stream(
                "POST",
                "/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise LocalModelError(
                            "Ollama lieferte einen ungültigen Streaming-Chunk."
                        ) from error
                    message = chunk.get("message", {})
                    if not isinstance(message, dict):
                        raise LocalModelError(
                            "Ollama lieferte einen ungültigen Streaming-Chunk."
                        )
                    content = message.get("content", "")
                    thinking = message.get("thinking", "")
                    if isinstance(content, str) and content:
                        content_parts.append(content)
                        if on_token is not None:
                            callback_result = on_token(content)
                            if inspect.isawaitable(callback_result):
                                await callback_result
                    if isinstance(thinking, str) and thinking:
                        thinking_parts.append(thinking)
                    for index, call in enumerate(message.get("tool_calls") or []):
                        if not isinstance(call, dict):
                            continue
                        current_call = tool_calls.setdefault(index, {})
                        current_function = current_call.setdefault("function", {})
                        function = call.get("function", {})
                        if isinstance(function, dict):
                            if isinstance(function.get("name"), str):
                                current_function["name"] = function["name"]
                            if "arguments" in function:
                                current_function["arguments"] = function["arguments"]
                        current_call["type"] = call.get("type", "function")
                    if chunk.get("done"):
                        done_reason = chunk.get("done_reason")
        except httpx.HTTPStatusError as error:
            detail = error.response.text.strip().replace("\n", " ")[:400]
            logger.error(
                "Ollama returned HTTP %d for streaming /api/chat. Response: %s",
                error.response.status_code,
                detail or "(empty response)",
            )
            raise LocalModelError(
                f"Ollama meldet HTTP {error.response.status_code} für /api/chat: "
                f"{detail or 'keine Details'}. Prüfe, ob das konfigurierte Modell "
                "in `ollama list` vorhanden ist."
            ) from error
        except (httpx.ConnectError, httpx.TimeoutException) as error:
            logger.exception("Cannot reach Ollama endpoint /api/chat.")
            raise LocalModelError(
                f"Ollama ist unter {self._client.base_url} nicht erreichbar "
                "(Endpoint /api/chat). Stelle sicher, dass `ollama serve` läuft "
                "und OLLAMA_HOST auf dieselbe Adresse zeigt."
            ) from error
        except httpx.HTTPError as error:
            logger.exception("Ollama streaming request failed.")
            raise LocalModelError(
                f"Die Streaming-Anfrage an Ollama ist fehlgeschlagen: {error}."
            ) from error
        return {
            "message": {
                "content": "".join(content_parts),
                "thinking": "".join(thinking_parts),
                "tool_calls": list(tool_calls.values()),
            },
            "done_reason": done_reason,
        }

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized_messages = []
        for message in messages:
            normalized_message = dict(message)
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                normalized_tool_calls = []
                for tool_call in tool_calls:
                    normalized_tool_call = dict(tool_call)
                    function = tool_call.get("function")
                    if isinstance(function, dict):
                        normalized_function = dict(function)
                        arguments = normalized_function.get("arguments")
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError as error:
                                raise LocalModelError(
                                    "Ein Ollama-Werkzeugaufruf enthält ungültige "
                                    "JSON-Argumente."
                                ) from error
                            if not isinstance(arguments, dict):
                                raise LocalModelError(
                                    "Ollama-Werkzeugargumente müssen ein JSON-Objekt sein."
                                )
                            normalized_function["arguments"] = arguments
                        normalized_tool_call["function"] = normalized_function
                    normalized_tool_calls.append(normalized_tool_call)
                normalized_message["tool_calls"] = normalized_tool_calls
            normalized_messages.append(normalized_message)
        return normalized_messages

    async def feature_extraction(
        self,
        text: str | list[str],
        *,
        normalize: bool | None = None,
        truncate: bool | None = None,
        **_: Any,
    ) -> list[list[float]]:
        texts = [text] if isinstance(text, str) else text
        response = await self._post(
            "/api/embed",
            {
                "model": self._model,
                "input": texts,
                "truncate": truncate if truncate is not None else True,
            },
        )
        embeddings = response.get("embeddings")
        if (
            not isinstance(embeddings, list)
            or len(embeddings) != len(texts)
            or not all(isinstance(vector, list) for vector in embeddings)
        ):
            raise LocalModelError(
                "Das lokale Embedding-Modell lieferte ungültige Vektoren."
            )
        return embeddings

    async def close(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=payload)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPStatusError as error:
            detail = error.response.text.strip().replace("\n", " ")[:400]
            logger.error(
                "Ollama returned HTTP %d for %s. Response: %s",
                error.response.status_code,
                path,
                detail or "(empty response)",
            )
            raise LocalModelError(
                f"Ollama meldet HTTP {error.response.status_code} für {path}: "
                f"{detail or 'keine Details'}. Prüfe, ob das konfigurierte Modell "
                "in `ollama list` vorhanden ist."
            ) from error
        except (httpx.ConnectError, httpx.TimeoutException) as error:
            logger.exception("Cannot reach Ollama endpoint %s.", path)
            raise LocalModelError(
                f"Ollama ist unter {self._client.base_url} nicht erreichbar "
                f"(Endpoint {path}). Stelle sicher, dass `ollama serve` läuft "
                "und OLLAMA_HOST auf dieselbe Adresse zeigt."
            ) from error
        except httpx.HTTPError as error:
            logger.exception("Ollama request failed at endpoint %s.", path)
            raise LocalModelError(
                f"Die Anfrage an Ollama {path} ist fehlgeschlagen: {error}."
            ) from error
        except ValueError as error:
            raise LocalModelError("Ollama lieferte eine ungültige JSON-Antwort.") from error
        if not isinstance(result, dict):
            raise LocalModelError("Ollama lieferte ein ungültiges Antwortformat.")
        return result
