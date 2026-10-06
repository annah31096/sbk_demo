"""Tracing setup and OpenInference span helpers."""

import json
from contextlib import contextmanager
from typing import Any, Iterator, Protocol

from opentelemetry import trace
from opentelemetry.trace import Span

TRACER = trace.get_tracer("sbk_demo")
CAPTURE_CONTENT = False


class TracerProvider(Protocol):
    def force_flush(self) -> bool: ...

    def shutdown(self) -> None: ...


@contextmanager
def trace_span(name: str, attributes: dict[str, str | int | bool]) -> Iterator[Span]:
    with TRACER.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span


def record_llm_request(span: Span, path: str, payload: dict[str, Any]) -> None:
    if not CAPTURE_CONTENT:
        return

    if path == "/api/chat":
        for message_index, message in enumerate(payload.get("messages", [])):
            prefix = f"llm.input_messages.{message_index}.message"
            role = message.get("role")
            if isinstance(role, str):
                span.set_attribute(f"{prefix}.role", role)
            content = message.get("content")
            if isinstance(content, str) and content:
                span.set_attribute(f"{prefix}.content", content)
            for call_index, tool_call in enumerate(message.get("tool_calls", [])):
                function = tool_call.get("function", {})
                call_prefix = f"{prefix}.tool_calls.{call_index}.tool_call"
                name = function.get("name")
                if isinstance(name, str):
                    span.set_attribute(f"{call_prefix}.function.name", name)
                arguments = function.get("arguments")
                if arguments is not None:
                    span.set_attribute(
                        f"{call_prefix}.function.arguments",
                        arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments, ensure_ascii=False),
                    )
        for tool_index, tool in enumerate(payload.get("tools", [])):
            function = tool.get("function", {})
            tool_name = function.get("name")
            if isinstance(tool_name, str):
                span.set_attribute(f"llm.tools.{tool_index}.tool.name", tool_name)
            span.set_attribute(
                f"llm.tools.{tool_index}.tool.json_schema",
                json.dumps(tool, ensure_ascii=False),
            )
    elif path == "/api/embed":
        texts = payload.get("input", [])
        if isinstance(texts, str):
            texts = [texts]
        for index, text in enumerate(texts):
            if isinstance(text, str):
                span.set_attribute(f"embedding.embeddings.{index}.embedding.text", text)


def record_llm_response(span: Span, path: str, response: dict[str, Any]) -> None:
    if path == "/api/chat":
        message = response.get("message")
        if isinstance(message, dict) and CAPTURE_CONTENT:
            prefix = "llm.output_messages.0.message"
            span.set_attribute(f"{prefix}.role", "assistant")
            content = message.get("content")
            if isinstance(content, str) and content:
                span.set_attribute(f"{prefix}.content", content)
            for call_index, tool_call in enumerate(message.get("tool_calls", [])):
                function = tool_call.get("function", {})
                call_prefix = f"{prefix}.tool_calls.{call_index}.tool_call"
                name = function.get("name")
                if isinstance(name, str):
                    span.set_attribute(f"{call_prefix}.function.name", name)
                arguments = function.get("arguments")
                if arguments is not None:
                    span.set_attribute(
                        f"{call_prefix}.function.arguments",
                        arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments, ensure_ascii=False),
                    )
    elif path == "/api/embed" and CAPTURE_CONTENT:
        vectors = response.get("embeddings", [])
        for index, vector in enumerate(vectors):
            if isinstance(vector, list):
                span.set_attribute(
                    f"embedding.embeddings.{index}.embedding.vector",
                    vector,
                )

    prompt_tokens = response.get("prompt_eval_count")
    completion_tokens = response.get("eval_count")
    if isinstance(prompt_tokens, int):
        span.set_attribute("llm.token_count.prompt", prompt_tokens)
    if isinstance(completion_tokens, int):
        span.set_attribute("llm.token_count.completion", completion_tokens)
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        span.set_attribute(
            "llm.token_count.total",
            prompt_tokens + completion_tokens,
        )
    done_reason = response.get("done_reason")
    if isinstance(done_reason, str):
        span.set_attribute("llm.finish_reason", done_reason)


def record_tool_input(span: Span, value: str) -> None:
    if CAPTURE_CONTENT:
        span.set_attribute("input.value", value)


def record_tool_output(span: Span, value: str) -> None:
    if CAPTURE_CONTENT:
        span.set_attribute("output.value", value)


def setup_phoenix_tracing(
    *,
    enabled: bool,
    endpoint: str,
    project_name: str,
    capture_content: bool,
) -> TracerProvider | None:
    global CAPTURE_CONTENT
    CAPTURE_CONTENT = capture_content
    if not enabled:
        return None

    from phoenix.otel import register

    return register(
        project_name=project_name,
        endpoint=endpoint,
        protocol="grpc",
        batch=True,
        auto_instrument=False,
        verbose=False,
    )


def shutdown_phoenix_tracing(tracing: TracerProvider | None) -> None:
    if tracing is None:
        return

    tracing.force_flush()
    tracing.shutdown()
