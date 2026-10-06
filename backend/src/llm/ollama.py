"""Async client for locally hosted Ollama chat and embedding models."""

import inspect
import json
import logging
from types import SimpleNamespace
from typing import Any, Callable
from uuid import uuid4

import httpx

from backend.config import DEFAULT_OLLAMA_TIMEOUT_SECONDS
from backend.src.observability.tracing import (
    record_llm_request,
    record_llm_response,
    trace_span,
)

logger = logging.getLogger(__name__)


class LocalModelError(RuntimeError):
    """Raised when the local Ollama runtime cannot serve a model request."""


class LocalOllamaClient:
    def __init__(
        self,
        host: str,
        model: str,
        *,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
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
        """
        Send a chat request to Ollama.

        Tool selection should normally use:
            stream=False
            think=False

        Final answer generation may use:
            stream=True
            think=True

        This separation makes structured tool calling considerably more
        reliable with local Ollama models.
        """

        options: dict[str, Any] = {}

        if max_tokens is not None:
            options["num_predict"] = max_tokens

        if temperature is not None:
            options["temperature"] = temperature

        normalized_messages = self._normalize_messages(messages)

        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": normalized_messages,
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

        logger.info(
            "Ollama chat request: model=%s stream=%s think=%s "
            "messages=%d tools=%d",
            payload["model"],
            stream,
            think,
            len(normalized_messages),
            len(tools or []),
        )

        if tools:
            logger.info(
                "Ollama tools: %s",
                ", ".join(
                    self._tool_name(tool)
                    for tool in tools
                ),
            )

        if stream:
            response = await self._stream_chat(
                payload,
                on_token,
            )
        else:
            response = await self._post(
                "/api/chat",
                payload,
            )

        return self._build_chat_response(response)

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------

    def _build_chat_response(
        self,
        response: dict[str, Any],
    ) -> Any:
        raw_message = response.get("message")

        if not isinstance(raw_message, dict):
            raise LocalModelError(
                "Das lokale Ollama-Modell lieferte keine gültige "
                "Antwortnachricht."
            )

        content = raw_message.get(
            "content",
            "",
        )

        if not isinstance(content, str):
            raise LocalModelError(
                "Das lokale Ollama-Modell lieferte ein ungültiges "
                "Content-Format."
            )

        thinking = raw_message.get(
            "thinking",
            "",
        )

        if not isinstance(thinking, str):
            thinking = ""

        raw_tool_calls = raw_message.get(
            "tool_calls",
            [],
        )

        if raw_tool_calls is None:
            raw_tool_calls = []

        if not isinstance(raw_tool_calls, list):
            raise LocalModelError(
                "Ollama lieferte ein ungültiges Tool-Call-Format."
            )

        tool_calls: list[SimpleNamespace] = []

        for index, raw_call in enumerate(raw_tool_calls):
            normalized = self._normalize_tool_call(
                raw_call,
                index,
            )

            if normalized is None:
                continue

            tool_calls.append(normalized)

        done_reason = response.get(
            "done_reason"
        )

        logger.info(
            "Ollama chat response: content=%d chars "
            "thinking=%d chars tools=%d done_reason=%s",
            len(content),
            len(thinking),
            len(tool_calls),
            done_reason,
        )

        for index, tool_call in enumerate(tool_calls):
            logger.info(
                "Ollama tool call #%d: name=%s arguments=%r",
                index,
                tool_call.function.name,
                tool_call.function.arguments,
            )

        message = SimpleNamespace(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
        )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message,
                )
            ],
            done_reason=done_reason,
        )

    @staticmethod
    def _normalize_tool_call(
        raw_call: Any,
        index: int,
    ) -> SimpleNamespace | None:
        if not isinstance(raw_call, dict):
            logger.warning(
                "Ignoring invalid Ollama tool call #%d: %r",
                index,
                raw_call,
            )
            return None

        function = raw_call.get(
            "function",
            {},
        )

        if not isinstance(function, dict):
            logger.warning(
                "Ignoring Ollama tool call #%d with invalid function.",
                index,
            )
            return None

        name = function.get(
            "name",
            "",
        )

        if not isinstance(name, str):
            name = str(name or "")

        arguments = function.get(
            "arguments",
            {},
        )

        # Ollama normally returns an object here. Some model/API
        # combinations may return a JSON string instead.
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                # Keep the original string. The agent layer can report
                # a useful tool argument error later.
                parsed_arguments = arguments
            else:
                arguments = parsed_arguments

        tool_id = raw_call.get("id")

        if not isinstance(tool_id, str) or not tool_id:
            tool_id = f"ollama-{uuid4().hex}"

        return SimpleNamespace(
            id=tool_id,
            type=raw_call.get(
                "type",
                "function",
            ),
            function=SimpleNamespace(
                name=name,
                arguments=arguments,
            ),
        )

    @staticmethod
    def _tool_name(tool: dict[str, Any]) -> str:
        if not isinstance(tool, dict):
            return "unknown"

        function = tool.get(
            "function",
            {},
        )

        if not isinstance(function, dict):
            return "unknown"

        name = function.get(
            "name",
            "unknown",
        )

        return str(name)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def _stream_chat(
        self,
        payload: dict[str, Any],
        on_token: Callable[[str], Any] | None,
    ) -> dict[str, Any]:
        with trace_span(
            "ollama.chat",
            {
                "openinference.span.kind": "LLM",
                "llm.system": "ollama",
                "llm.model_name": payload["model"],
                "llm.request.model_name": payload["model"],
                "llm.request.streaming": True,
            },
        ) as span:
            record_llm_request(
                span,
                "/api/chat",
                payload,
            )

            response = await self._stream_chat_request(
                payload,
                on_token,
            )

            record_llm_response(
                span,
                "/api/chat",
                response,
            )

            return response

    async def _stream_chat_request(
        self,
        payload: dict[str, Any],
        on_token: Callable[[str], Any] | None,
    ) -> dict[str, Any]:
        content_parts: list[str] = []
        thinking_parts: list[str] = []

        # Tool calls are indexed because Ollama can send multiple
        # tool calls in one assistant response.
        tool_calls: dict[int, dict[str, Any]] = {}

        done_reason: str | None = None
        prompt_eval_count: int | None = None
        eval_count: int | None = None
        response_model: str | None = None

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
                            "Ollama lieferte einen ungültigen "
                            "Streaming-Chunk."
                        ) from error

                    if not isinstance(chunk, dict):
                        raise LocalModelError(
                            "Ollama lieferte ein ungültiges "
                            "Streaming-Chunk-Format."
                        )

                    message = chunk.get(
                        "message",
                        {},
                    )

                    if not isinstance(message, dict):
                        raise LocalModelError(
                            "Ollama lieferte eine ungültige "
                            "Nachricht im Streaming-Chunk."
                        )

                    # --------------------------------------------------
                    # Content
                    # --------------------------------------------------

                    content = message.get(
                        "content",
                        "",
                    )

                    if isinstance(content, str) and content:
                        content_parts.append(content)

                        if on_token is not None:
                            callback_result = on_token(
                                content
                            )

                            if inspect.isawaitable(
                                callback_result
                            ):
                                await callback_result

                    # --------------------------------------------------
                    # Thinking
                    # --------------------------------------------------

                    thinking = message.get(
                        "thinking",
                        "",
                    )

                    if isinstance(thinking, str) and thinking:
                        thinking_parts.append(thinking)

                    # --------------------------------------------------
                    # Tool calls
                    # --------------------------------------------------

                    raw_tool_calls = message.get(
                        "tool_calls",
                        [],
                    )

                    if not isinstance(
                        raw_tool_calls,
                        list,
                    ):
                        raw_tool_calls = []

                    for index, raw_call in enumerate(
                        raw_tool_calls
                    ):
                        if not isinstance(
                            raw_call,
                            dict,
                        ):
                            continue

                        current = tool_calls.setdefault(
                            index,
                            {
                                "type": "function",
                                "function": {},
                            },
                        )

                        call_type = raw_call.get(
                            "type"
                        )

                        if isinstance(
                            call_type,
                            str,
                        ):
                            current["type"] = call_type

                        function = raw_call.get(
                            "function",
                            {},
                        )

                        if not isinstance(
                            function,
                            dict,
                        ):
                            continue

                        name = function.get(
                            "name"
                        )

                        if isinstance(
                            name,
                            str,
                        ) and name:
                            current[
                                "function"
                            ]["name"] = name

                        if "arguments" in function:
                            current[
                                "function"
                            ]["arguments"] = function[
                                "arguments"
                            ]

                    # --------------------------------------------------
                    # Final chunk
                    # --------------------------------------------------

                    if chunk.get("done"):
                        done_reason = chunk.get(
                            "done_reason"
                        )

                        prompt_eval_count = chunk.get(
                            "prompt_eval_count"
                        )

                        eval_count = chunk.get(
                            "eval_count"
                        )

                        response_model = chunk.get(
                            "model"
                        )

        except httpx.HTTPStatusError as error:
            detail = (
                error.response.text
                .strip()
                .replace("\n", " ")[:400]
            )

            logger.error(
                "Ollama returned HTTP %d for "
                "streaming /api/chat. Response: %s",
                error.response.status_code,
                detail or "(empty response)",
            )

            raise LocalModelError(
                f"Ollama meldet HTTP "
                f"{error.response.status_code} für /api/chat: "
                f"{detail or 'keine Details'}."
            ) from error

        except (
            httpx.ConnectError,
            httpx.TimeoutException,
        ) as error:
            logger.exception(
                "Cannot reach Ollama endpoint /api/chat."
            )

            raise LocalModelError(
                f"Ollama ist unter "
                f"{self._client.base_url} "
                "nicht erreichbar. Stelle sicher, "
                "dass `ollama serve` läuft."
            ) from error

        except httpx.HTTPError as error:
            logger.exception(
                "Ollama streaming request failed."
            )

            raise LocalModelError(
                f"Die Streaming-Anfrage an Ollama "
                f"ist fehlgeschlagen: {error}."
            ) from error

        result = {
            "message": {
                "content": "".join(
                    content_parts
                ),
                "thinking": "".join(
                    thinking_parts
                ),
                "tool_calls": list(
                    tool_calls.values()
                ),
            },
            "done_reason": done_reason,
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "model": response_model,
        }

        logger.info(
            "Ollama stream finished: "
            "content=%d chars, "
            "thinking=%d chars, "
            "tool_calls=%d, "
            "done_reason=%s",
            len(
                result["message"]["content"]
            ),
            len(
                result["message"]["thinking"]
            ),
            len(
                result["message"]["tool_calls"]
            ),
            done_reason,
        )

        return result

    # ------------------------------------------------------------------
    # Message normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized_messages: list[dict[str, Any]] = []

        for message in messages:
            if not isinstance(
                message,
                dict,
            ):
                raise LocalModelError(
                    "Eine Chat-Nachricht hat ein ungültiges Format."
                )

            normalized_message = dict(message)

            tool_calls = message.get(
                "tool_calls"
            )

            if isinstance(
                tool_calls,
                list,
            ):
                normalized_tool_calls: list[
                    dict[str, Any]
                ] = []

                for tool_call in tool_calls:
                    if not isinstance(
                        tool_call,
                        dict,
                    ):
                        continue

                    normalized_tool_call = dict(
                        tool_call
                    )

                    function = tool_call.get(
                        "function"
                    )

                    if isinstance(
                        function,
                        dict,
                    ):
                        normalized_function = dict(
                            function
                        )

                        arguments = (
                            normalized_function.get(
                                "arguments"
                            )
                        )

                        if isinstance(
                            arguments,
                            str,
                        ):
                            try:
                                arguments = json.loads(
                                    arguments
                                )
                            except json.JSONDecodeError as error:
                                raise LocalModelError(
                                    "Ein Ollama-Werkzeugaufruf "
                                    "enthält ungültige "
                                    "JSON-Argumente."
                                ) from error

                            if not isinstance(
                                arguments,
                                dict,
                            ):
                                raise LocalModelError(
                                    "Ollama-Werkzeugargumente "
                                    "müssen ein JSON-Objekt sein."
                                )

                            normalized_function[
                                "arguments"
                            ] = arguments

                        normalized_tool_call[
                            "function"
                        ] = normalized_function

                    normalized_tool_calls.append(
                        normalized_tool_call
                    )

                normalized_message[
                    "tool_calls"
                ] = normalized_tool_calls

            normalized_messages.append(
                normalized_message
            )

        return normalized_messages

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def feature_extraction(
        self,
        text: str | list[str],
        *,
        normalize: bool | None = None,
        truncate: bool | None = None,
        **_: Any,
    ) -> list[list[float]]:
        texts = (
            [text]
            if isinstance(text, str)
            else text
        )

        response = await self._post(
            "/api/embed",
            {
                "model": self._model,
                "input": texts,
                "truncate": (
                    truncate
                    if truncate is not None
                    else True
                ),
            },
        )

        embeddings = response.get(
            "embeddings"
        )

        if (
            not isinstance(
                embeddings,
                list,
            )
            or len(embeddings) != len(texts)
            or not all(
                isinstance(
                    vector,
                    list,
                )
                for vector in embeddings
            )
        ):
            raise LocalModelError(
                "Das lokale Embedding-Modell "
                "lieferte ungültige Vektoren."
            )

        return embeddings

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        span_name = (
            "ollama.embeddings"
            if path == "/api/embed"
            else "ollama.chat"
        )

        span_kind = (
            "EMBEDDING"
            if path == "/api/embed"
            else "LLM"
        )

        with trace_span(
            span_name,
            {
                "openinference.span.kind": span_kind,
                "llm.system": "ollama",
                "llm.model_name": payload["model"],
                "llm.request.model_name": payload["model"],
            },
        ) as span:
            record_llm_request(
                span,
                path,
                payload,
            )

            response = await self._post_request(
                path,
                payload,
            )

            record_llm_response(
                span,
                path,
                response,
            )

            return response

    async def _post_request(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = await self._client.post(
                path,
                json=payload,
            )

            response.raise_for_status()

            result = response.json()

        except httpx.HTTPStatusError as error:
            detail = (
                error.response.text
                .strip()
                .replace("\n", " ")[:400]
            )

            logger.error(
                "Ollama returned HTTP %d for %s. "
                "Response: %s",
                error.response.status_code,
                path,
                detail or "(empty response)",
            )

            raise LocalModelError(
                f"Ollama meldet HTTP "
                f"{error.response.status_code} für {path}: "
                f"{detail or 'keine Details'}."
            ) from error

        except (
            httpx.ConnectError,
            httpx.TimeoutException,
        ) as error:
            logger.exception(
                "Cannot reach Ollama endpoint %s.",
                path,
            )

            raise LocalModelError(
                f"Ollama ist unter "
                f"{self._client.base_url} "
                f"nicht erreichbar "
                f"(Endpoint {path}). Stelle sicher, "
                "dass `ollama serve` läuft."
            ) from error

        except httpx.HTTPError as error:
            logger.exception(
                "Ollama request failed at endpoint %s.",
                path,
            )

            raise LocalModelError(
                f"Die Anfrage an Ollama {path} "
                f"ist fehlgeschlagen: {error}."
            ) from error

        except ValueError as error:
            raise LocalModelError(
                "Ollama lieferte eine ungültige "
                "JSON-Antwort."
            ) from error

        if not isinstance(
            result,
            dict,
        ):
            raise LocalModelError(
                "Ollama lieferte ein ungültiges "
                "Antwortformat."
            )

        return result
