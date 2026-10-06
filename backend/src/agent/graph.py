"""
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import operator
import re
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, AsyncIterator, Callable, TypedDict
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import END, START, StateGraph

from backend.config import BackendConfig

from backend.src.knowledge.retriever import MarkdownRetriever
from backend.src.llm.ollama import LocalOllamaClient
from backend.src.observability.tracing import (
    record_tool_input,
    record_tool_output,
    trace_span,
)
from backend.src.prompts.loader import load_prompt
from backend.src.tools.calculator import calculate


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = load_prompt("agent/system.txt")
RAG_SYSTEM_PROMPT = load_prompt("rag/rag_system.txt")
CALCULATE_TOOL_PROMPT = load_prompt("tools/calculate_tool.txt")
CALCULATE_TOOL_INPUT_PROMPT = load_prompt("tools/calculate_tool_input.txt")
SEARCH_KNOWLEDGE_TOOL_PROMPT = load_prompt("tools/search_knowledge_tool.txt")
SEARCH_KNOWLEDGE_INPUT_PROMPT = load_prompt("tools/search_knowledge_input.txt")


# ============================================================================
# Tools
# ============================================================================

CALCULATE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": CALCULATE_TOOL_PROMPT,
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": CALCULATE_TOOL_INPUT_PROMPT,
                }
            },
            "required": ["expression"],
        },
    },
}


SEARCH_KNOWLEDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": SEARCH_KNOWLEDGE_TOOL_PROMPT,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": SEARCH_KNOWLEDGE_INPUT_PROMPT,
                }
            },
            "required": ["query"],
        },
    },
}


TOOLS = [
    CALCULATE_TOOL,
    SEARCH_KNOWLEDGE_TOOL,
]


TOOL_NAMES = {
    "calculate",
    "search_knowledge_base",
}


# ============================================================================
# Replies
# ============================================================================

NO_SOURCE_REPLY = load_prompt("responses/no_source_reply.txt")
TOOL_ERROR_REPLY = load_prompt("responses/tool_error_reply.txt")
PLANNING_ERROR_REPLY = load_prompt("responses/planning_error_reply.txt")
CALCULATION_REQUIRED_REPLY = load_prompt("responses/calculation_required_reply.txt")


# ============================================================================
# State
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]

    tool_call_count: Annotated[
        int,
        operator.add,
    ]

    tool_evidence: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]

    trace_events: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]

    plan: list[dict[str, Any]]

    plan_step_index: int

    planning_attempts: int

    rag_documents: list[dict[str, Any]]

    rag_context: str

    calculation_result: str

    calculation_expression: str

    final_reply: str

    control_action: str

# ============================================================================
# Generic helpers
# ============================================================================

def _value(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(
        obj,
        key,
        default,
    )


def _function_value(
    tool_call: Any,
    key: str,
    default: Any = None,
) -> Any:
    function = _value(
        tool_call,
        "function",
        {},
    )

    return _value(
        function,
        key,
        default,
    )


def _clean_json(
    value: str,
) -> str:
    text = value.strip()

    # Markdown code fences entfernen.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


def _extract_json_object(
    value: str,
) -> str:
    """Extrahiert das erste brauchbare JSON-Objekt aus Modelltext."""

    cleaned = _clean_json(value)

    # Direkt versuchen.
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    start = cleaned.find("{")

    if start < 0:
        raise ValueError(
            "Kein JSON-Objekt in Planner-Antwort gefunden."
        )

    depth = 0
    in_string = False
    escaped = False

    for index in range(
        start,
        len(cleaned),
    ):
        char = cleaned[index]

        if escaped:
            escaped = False
            continue

        if char == "\\" and in_string:
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                return cleaned[
                    start:index + 1
                ]

    raise ValueError(
        "JSON-Objekt ist unvollständig."
    )


def _safe_json_loads(
    value: Any,
) -> Any:
    if isinstance(
        value,
        (dict, list),
    ):
        return value

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            "Ungültiges JSON."
        )

    extracted = _extract_json_object(
        value
    )

    return json.loads(
        extracted
    )


def _latest_user_question(
    state: AgentState,
) -> str:
    for message in reversed(
        state["messages"]
    ):
        if message.get("role") == "user":
            content = message.get(
                "content",
                "",
            )

            if isinstance(
                content,
                str,
            ):
                return content

    return ""


# ============================================================================
# Deterministic intent detection
# ============================================================================

CALCULATION_PATTERNS = (
    r"\bberechne\b",
    r"\bberechnen\b",
    r"\bberechnung\b",
    r"\bausrechnen\b",
    r"\bwie\s+viel\b",
    r"\bwieviel\b",
    r"\bwelcher\s+betrag\b",
    r"\bwelchen\s+betrag\b",
    r"\bbetrag\b",
    r"\bsumme\b",
    r"\bkosten\b",
    r"\bprozent\b",
    r"\bprozentsatz\b",
    r"\banteil\b",
    r"\bgrenze\b.*\b(?:euro|€|betrag|einkommen)\b",
    r"%"
)


KNOWLEDGE_PATTERNS = (
    r"\bregel\b",
    r"\bregeln\b",
    r"\bvoraussetzung\b",
    r"\bvoraussetzungen\b",
    r"\banspruch\b",
    r"\banspruchsberechtigt\b",
    r"\bbelastungsgrenze\b",
    r"\bzuzahlung\b",
    r"\beinkommen\b",
    r"\bgrenze\b",
    r"\bsatz\b",
    r"\bprozentsatz\b",
    r"\bgesetz\b",
    r"\brichtlinie\b",
    r"\bverordnung\b",
    r"\bregelung\b",
    r"\bversorgung\b",
    r"\bleistung\b",
    r"\bkrankenkasse\b",
    r"\banlage\b",
    r"\bdokument\b",
)


def _contains_pattern(
    text: str,
    patterns: tuple[str, ...],
) -> bool:
    lowered = text.lower()

    return any(
        re.search(
            pattern,
            lowered,
        )
        for pattern in patterns
    )


def _calculation_requested(
    text: str,
) -> bool:
    return _contains_pattern(
        text,
        CALCULATION_PATTERNS,
    )


def _knowledge_required(
    text: str,
) -> bool:
    return _contains_pattern(
        text,
        KNOWLEDGE_PATTERNS,
    )


def _looks_like_numeric_question(
    text: str,
) -> bool:
    if _calculation_requested(text):
        return True

    return bool(
        re.search(
            r"\d+(?:[.,]\d+)?\s*(?:€|euro|%|prozent)?",
            text.lower(),
        )
        and re.search(
            r"\b(?:wie|welcher|welche|berech|betrag|grenze|kosten)\b",
            text.lower(),
        )
    )


# ============================================================================
# Callback
# ============================================================================

class BackendLangChainCallbackHandler(
    BaseCallbackHandler
):
    """Log lifecycle events without logging prompts/output."""

    def __init__(self) -> None:
        self._started: dict[
            UUID,
            tuple[str, float],
        ] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if (
            metadata
            and metadata.get(
                "langgraph_node"
            )
        ):
            return

        identifier = (
            (serialized or {}).get("name")
            or (serialized or {}).get("id")
        )

        if isinstance(
            identifier,
            list,
        ):
            name = (
                str(identifier[-1])
                if identifier
                else "chain"
            )
        else:
            name = (
                str(identifier)
                if identifier
                else "chain"
            )

        self._started[run_id] = (
            name,
            perf_counter(),
        )

        logger.info(
            "LangChain chain started: %s",
            name,
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        started = self._started.pop(
            run_id,
            None,
        )

        if started is None:
            return

        name, started_at = started

        logger.info(
            "LangChain chain completed: %s (%.0f ms)",
            name,
            (perf_counter() - started_at) * 1000,
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        started = self._started.pop(
            run_id,
            None,
        )

        if started is None:
            return

        logger.error(
            "LangChain chain failed: %s (%s)",
            started[0],
            type(error).__name__,
        )


# ============================================================================
# Agent
# ============================================================================

class ChatAgent:

    def __init__(
        self,
        model: str | None = None,
        embedding_model: str | None = None,
        keyword_model: str | None = None,
        ollama_host: str | None = None,
        documents_directory: Path | None = None,
        retriever: MarkdownRetriever | None = None,
        chat_client: Any | None = None,
        config: BackendConfig | None = None,
    ) -> None:
        self._config = config or BackendConfig.from_env()
        model = model or self._config.chat_model
        embedding_model = embedding_model or self._config.embedding_model
        keyword_model = keyword_model or self._config.keyword_model
        ollama_host = ollama_host or self._config.ollama_host
        self._model = model

        self._client = (
            chat_client
            or LocalOllamaClient(
                ollama_host,
                model,
                timeout=self._config.ollama_timeout_seconds,
            )
        )

        self._retriever = (
            retriever
            or MarkdownRetriever(
                documents_directory
                or self._config.knowledge_directory,
                embedding_model,
                keyword_model=keyword_model,
                ollama_host=ollama_host,
                top_k=self._config.rag_top_k,
                minimum_score=self._config.rag_minimum_score,
                embedding_batch_size=self._config.rag_embedding_batch_size,
                keyword_max_tokens=self._config.keyword_generation_max_tokens,
                keyword_temperature=self._config.keyword_generation_temperature,
                index_directory=self._config.rag_index_directory,
                ollama_timeout=self._config.ollama_timeout_seconds,
            )
        )

        graph = StateGraph(
            AgentState
        )

        graph.add_node(
            "plan",
            self._logged_node(
                "plan",
                self._plan,
            ),
        )

        graph.add_node(
            "execute_plan_step",
            self._logged_node(
                "execute_plan_step",
                self._execute_plan_step,
            ),
        )

        graph.add_node(
            "finalize",
            self._logged_node(
                "finalize",
                self._finalize,
            ),
        )

        graph.add_edge(
            START,
            "plan",
        )

        graph.add_conditional_edges(
            "plan",
            self._after_plan,
            {
                "execute": "execute_plan_step",
                "final": "finalize",
            },
        )

        graph.add_conditional_edges(
            "execute_plan_step",
            self._after_step,
            {
                "execute": "execute_plan_step",
                "plan": "plan",
                "final": "finalize",
            },
        )

        graph.add_edge(
            "finalize",
            END,
        )

        self._graph = graph.compile()

    # ========================================================================
    # Node wrapper
    # ========================================================================

    @staticmethod
    def _logged_node(
        name: str,
        node: Callable[
            [AgentState],
            Any,
        ],
    ) -> Callable[
        [AgentState],
        Any,
    ]:

        async def invoke(
            state: AgentState,
        ) -> Any:
            started = perf_counter()

            logger.info(
                "LangGraph node started: %s",
                name,
            )

            try:
                result = node(state)

                if inspect.isawaitable(
                    result
                ):
                    result = await result

                return result

            except Exception:
                logger.exception(
                    "LangGraph node failed: %s",
                    name,
                )
                raise

            finally:
                logger.info(
                    "LangGraph node completed: %s (%.0f ms)",
                    name,
                    (
                        perf_counter()
                        - started
                    )
                    * 1000,
                )

        return invoke

    # ========================================================================
    # Lifecycle
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        await self._retriever.initialize()

    async def close(
        self,
    ) -> None:
        await self._client.close()
        await self._retriever.close()

    # ========================================================================
    # Streaming
    # ========================================================================

    async def stream(
        self,
        message: str,
        history: list[
            dict[str, str]
        ]
        | None = None,
    ) -> AsyncIterator[
        dict[str, Any]
    ]:

        event_queue: asyncio.Queue[
            Any
        ] = asyncio.Queue()

        conversation = [
            {
                "role": item["role"],
                "content": item["content"],
            }
            for item in (
                history or []
            )
            if (
                item.get("role")
                in {
                    "user",
                    "assistant",
                }
                and isinstance(
                    item.get("content"),
                    str,
                )
            )
        ]

        conversation.append(
            {
                "role": "user",
                "content": message,
            }
        )

        initial_state: AgentState = {
            "messages": conversation,
            "tool_call_count": 0,
            "tool_evidence": [],
            "trace_events": [],
            "plan": [],
            "plan_step_index": 0,
            "planning_attempts": 0,
            "rag_documents": [],
            "rag_context": "",
            "calculation_result": "",
            "calculation_expression": "",
            "final_reply": "",
            "control_action": "",
        }

        async def run_graph() -> None:
            try:
                async for event in self._graph.astream(
                    initial_state,
                    stream_mode="updates",
                    version="v2",
                    config={
                        "callbacks": [
                            BackendLangChainCallbackHandler()
                        ]
                    },
                ):
                    if not isinstance(
                        event,
                        dict,
                    ):
                        continue

                    if event.get(
                        "type"
                    ) != "updates":
                        continue

                    updates = event.get(
                        "data"
                    )

                    if not isinstance(
                        updates,
                        dict,
                    ):
                        continue

                    for node_update in updates.values():
                        if not isinstance(
                            node_update,
                            dict,
                        ):
                            continue

                        for trace_event in node_update.get(
                            "trace_events",
                            [],
                        ):
                            if isinstance(
                                trace_event,
                                dict,
                            ):
                                event_queue.put_nowait(
                                    trace_event
                                )

            except Exception as error:
                event_queue.put_nowait(
                    error
                )

            finally:
                event_queue.put_nowait(
                    None
                )

        graph_task = asyncio.create_task(
            run_graph()
        )

        try:
            while True:
                event = await event_queue.get()

                if event is None:
                    break

                if isinstance(
                    event,
                    Exception,
                ):
                    raise event

                yield event

            await graph_task

        finally:
            if not graph_task.done():
                graph_task.cancel()

            await asyncio.gather(
                graph_task,
                return_exceptions=True,
            )

    # ========================================================================
    # Planning prompt
    # ========================================================================

    def _planning_messages(
        self,
        state: AgentState,
    ) -> list[
        dict[str, Any]
    ]:

        prompt = load_prompt(
            "agent/planning.txt",
            rag_context=state.get("rag_context", ""),
            calculation_result=state.get("calculation_result", ""),
        )

        return [
            {
                "role": "system",
                "content": prompt,
            },
            *state["messages"],
        ]

    # ========================================================================
    # Planning
    # ========================================================================

    async def _plan(
        self,
        state: AgentState,
    ) -> dict[str, Any]:

        attempts = state.get(
            "planning_attempts",
            0,
        )

        user_question = _latest_user_question(
            state
        )

        # --------------------------------------------------------------------
        # Deterministic analysis
        # --------------------------------------------------------------------

        calculation_required = (
            _calculation_requested(
                user_question
            )
            or _looks_like_numeric_question(
                user_question
            )
        )

        knowledge_required = (
            _knowledge_required(
                user_question
            )
        )

        rag_already_available = bool(
            state.get(
                "rag_documents"
            )
        )

        calculation_already_available = bool(
            state.get(
                "calculation_result"
            )
        )

        logger.info(
            "Intent analysis: calculation=%s knowledge=%s "
            "rag_available=%s calculation_available=%s",
            calculation_required,
            knowledge_required,
            rag_already_available,
            calculation_already_available,
        )

        # --------------------------------------------------------------------
        # First planning call
        # --------------------------------------------------------------------

        if (
            not rag_already_available
            and not calculation_already_available
        ):
            if attempts < self._config.max_planning_attempts:
                try:
                    response = await self._client.chat_completion(
                        messages=self._planning_messages(
                            state
                        ),
                        max_tokens=self._config.planning_max_tokens,
                        temperature=self._config.agent_temperature,
                        think=False,
                        stream=False,
                    )

                    if response.choices:
                        content = _value(
                            response.choices[0].message,
                            "content",
                            "",
                        )

                        if not isinstance(
                            content,
                            str,
                        ):
                            content = str(
                                content or ""
                            )

                        logger.info(
                            "Planner response: %s",
                            content[:1500],
                        )

                        parsed = _safe_json_loads(
                            content
                        )

                        steps = self._parse_plan(
                            parsed
                        )

                        if steps:
                            steps = self._normalize_plan(
                                steps,
                                state,
                            )

                            logger.info(
                                "Normalized execution plan: %s",
                                json.dumps(
                                    steps,
                                    ensure_ascii=False,
                                ),
                            )

                            return {
                                "plan": steps,
                                "plan_step_index": 0,
                                "planning_attempts": (
                                    attempts + 1
                                ),
                                "trace_events": [
                                    {
                                        "type": "plan",
                                        "steps": steps,
                                    }
                                ],
                            }

                except Exception:
                    logger.exception(
                        "Planner response invalid. "
                        "Using deterministic fallback."
                    )

        # --------------------------------------------------------------------
        # HARD FALLBACK
        #
        # This is deliberately deterministic.
        # A bad local model can therefore NEVER prevent calculate from
        # running when the user asks for a calculation.
        # --------------------------------------------------------------------

        fallback_plan = self._build_deterministic_plan(
            state,
            calculation_required=calculation_required,
            knowledge_required=knowledge_required,
        )

        logger.info(
            "Deterministic fallback plan: %s",
            json.dumps(
                fallback_plan,
                ensure_ascii=False,
            ),
        )

        return {
            "plan": fallback_plan,
            "plan_step_index": 0,
            "planning_attempts": attempts + 1,
            "trace_events": [
                {
                    "type": "plan",
                    "steps": fallback_plan,
                    "source": "deterministic_fallback",
                }
            ],
        }

    def _parse_plan(
        self,
        parsed: Any,
    ) -> list[
        dict[str, Any]
    ]:

        if not isinstance(
            parsed,
            dict,
        ):
            return []

        raw_steps = parsed.get(
            "steps",
            [],
        )

        if not isinstance(
            raw_steps,
            list,
        ):
            return []

        steps: list[
            dict[str, Any]
        ] = []

        for raw_step in raw_steps:
            if not isinstance(
                raw_step,
                dict,
            ):
                continue

            step_type = raw_step.get(
                "type"
            )

            if step_type == "search":
                query = raw_step.get(
                    "query",
                    "",
                )

                if (
                    isinstance(
                        query,
                        str,
                    )
                    and query.strip()
                ):
                    steps.append(
                        {
                            "type": "search",
                            "query": query.strip(),
                        }
                    )

            elif step_type == "calculate":
                expression = raw_step.get(
                    "expression",
                    "",
                )

                if not isinstance(
                    expression,
                    str,
                ):
                    expression = ""

                steps.append(
                    {
                        "type": "calculate",
                        "expression": expression.strip(),
                        "depends_on_search": bool(
                            raw_step.get(
                                "depends_on_search",
                                False,
                            )
                        ),
                    }
                )

            elif step_type == "answer":
                steps.append(
                    {
                        "type": "answer",
                    }
                )

        return steps

    # ========================================================================
    # Deterministic plan
    # ========================================================================

    def _build_deterministic_plan(
        self,
        state: AgentState,
        *,
        calculation_required: bool,
        knowledge_required: bool,
    ) -> list[
        dict[str, Any]
    ]:

        rag_available = bool(
            state.get(
                "rag_documents"
            )
        )

        calculation_available = bool(
            state.get(
                "calculation_result"
            )
        )

        steps: list[
            dict[str, Any]
        ] = []

        # --------------------------------------------------------------
        # RAG is required if knowledge is required.
        # --------------------------------------------------------------

        if knowledge_required and not rag_available:
            steps.append(
                {
                    "type": "search",
                    "query": (
                        _latest_user_question(
                            state
                        )
                    ),
                }
            )

        # --------------------------------------------------------------
        # Calculation is mandatory for numerical requests.
        # --------------------------------------------------------------

        if (
            calculation_required
            and not calculation_available
        ):
            steps.append(
                {
                    "type": "calculate",
                    "expression": "",
                    "depends_on_search": (
                        knowledge_required
                        or rag_available
                    ),
                }
            )

        steps.append(
            {
                "type": "answer",
            }
        )

        return self._normalize_plan(
            steps,
            state,
        )

    # ========================================================================
    # Plan normalization
    # ========================================================================

    def _normalize_plan(
        self,
        steps: list[
            dict[str, Any]
        ],
        state: AgentState,
    ) -> list[
        dict[str, Any]
    ]:

        user_question = _latest_user_question(
            state
        )

        calculation_required = (
            _calculation_requested(
                user_question
            )
            or _looks_like_numeric_question(
                user_question
            )
        )

        knowledge_required = _knowledge_required(
            user_question
        )

        rag_available = bool(
            state.get(
                "rag_documents"
            )
        )

        calculation_available = bool(
            state.get(
                "calculation_result"
            )
        )

        normalized: list[
            dict[str, Any]
        ] = []

        # --------------------------------------------------------------
        # First remove malformed steps.
        # --------------------------------------------------------------

        for step in steps:
            if not isinstance(
                step,
                dict,
            ):
                continue

            step_type = step.get(
                "type"
            )

            if step_type == "search":
                query = step.get(
                    "query",
                    "",
                )

                if isinstance(
                    query,
                    str,
                ) and query.strip():
                    normalized.append(
                        {
                            "type": "search",
                            "query": query.strip(),
                        }
                    )

            elif step_type == "calculate":
                expression = step.get(
                    "expression",
                    "",
                )

                if not isinstance(
                    expression,
                    str,
                ):
                    expression = ""

                normalized.append(
                    {
                        "type": "calculate",
                        "expression": expression.strip(),
                        "depends_on_search": bool(
                            step.get(
                                "depends_on_search",
                                False,
                            )
                        ),
                    }
                )

        # --------------------------------------------------------------
        # If knowledge is required, force search.
        # --------------------------------------------------------------

        has_search = any(
            step["type"] == "search"
            for step in normalized
        )

        if (
            knowledge_required
            and not rag_available
            and not has_search
        ):
            logger.warning(
                "Forcing search step."
            )

            normalized.insert(
                0,
                {
                    "type": "search",
                    "query": user_question,
                },
            )

            has_search = True

        # --------------------------------------------------------------
        # If calculation is required, force calculate.
        # --------------------------------------------------------------

        has_calculate = any(
            step["type"] == "calculate"
            for step in normalized
        )

        if (
            calculation_required
            and not calculation_available
            and not has_calculate
        ):
            logger.warning(
                "Planner omitted calculate. "
                "Forcing calculate step."
            )

            normalized.append(
                {
                    "type": "calculate",
                    "expression": "",
                    "depends_on_search": (
                        knowledge_required
                        or rag_available
                    ),
                }
            )

        # --------------------------------------------------------------
        # If RAG is already available and calculation exists, calculation
        # must use RAG values if the expression is not already concrete.
        # --------------------------------------------------------------

        if rag_available:
            for step in normalized:
                if step["type"] == "calculate":
                    if not step.get(
                        "expression"
                    ):
                        step[
                            "depends_on_search"
                        ] = True

        # --------------------------------------------------------------
        # Reorder:
        #
        # search -> calculate -> answer
        #
        # This is intentionally deterministic.
        # --------------------------------------------------------------

        searches = [
            step
            for step in normalized
            if step["type"] == "search"
        ]

        calculations = [
            step
            for step in normalized
            if step["type"] == "calculate"
        ]

        result = [
            *searches,
            *calculations,
            {
                "type": "answer",
            },
        ]

        # --------------------------------------------------------------
        # If search already happened and no further search is required,
        # don't search again.
        # --------------------------------------------------------------

        if rag_available:
            result = [
                step
                for step in result
                if step["type"] != "search"
            ]

            calculations = [
                step
                for step in result
                if step["type"] == "calculate"
            ]

            result = [
                *calculations,
                {
                    "type": "answer",
                },
            ]

        # --------------------------------------------------------------
        # If calculation already happened, don't repeat it.
        # --------------------------------------------------------------

        if calculation_available:
            result = [
                step
                for step in result
                if step["type"] != "calculate"
            ]

            result.append(
                {
                    "type": "answer",
                }
            )

        if len(result) > self._config.max_plan_steps:
            result = result[
                :self._config.max_plan_steps - 1
            ]

            result.append(
                {
                    "type": "answer",
                }
            )

        return result

    # ========================================================================
    # Plan routing
    # ========================================================================

    @staticmethod
    def _after_plan(
        state: AgentState,
    ) -> str:

        if state.get(
            "control_action"
        ):
            return "final"

        plan = state.get(
            "plan",
            [],
        )

        index = state.get(
            "plan_step_index",
            0,
        )

        if index >= len(plan):
            return "final"

        return "execute"

    @staticmethod
    def _after_step(
        state: AgentState,
    ) -> str:

        if state.get(
            "control_action"
        ):
            return "final"

        plan = state.get(
            "plan",
            [],
        )

        index = state.get(
            "plan_step_index",
            0,
        )

        if index >= len(plan):
            return "final"

        next_step = plan[index]

        # --------------------------------------------------------------
        # After RAG, re-plan.
        # --------------------------------------------------------------

        if (
            next_step["type"]
            == "calculate"
        ):
            if next_step.get(
                "depends_on_search"
            ):
                if state.get(
                    "rag_documents"
                ):
                    return "plan"

        return "execute"

    # ========================================================================
    # Execute current plan step
    # ========================================================================

    async def _execute_plan_step(
        self,
        state: AgentState,
    ) -> dict[str, Any]:

        plan = state.get(
            "plan",
            [],
        )

        index = state.get(
            "plan_step_index",
            0,
        )

        if index >= len(plan):
            return {}

        step = plan[index]
        if (
            step["type"] in {"search", "calculate"}
            and state.get("tool_call_count", 0)
            >= self._config.max_tool_calls_per_turn
        ):
            logger.warning(
                "Agent reached the per-turn tool-call limit (%d).",
                self._config.max_tool_calls_per_turn,
            )
            return {
                "control_action": "tool_error",
                "final_reply": TOOL_ERROR_REPLY,
                "plan_step_index": index + 1,
            }

        logger.info(
            "Executing plan step %d/%d: %s",
            index + 1,
            len(plan),
            step,
        )

        # --------------------------------------------------------------------
        # SEARCH
        # --------------------------------------------------------------------

        if step["type"] == "search":

            try:
                result = await self._execute_search(
                    step
                )
            except Exception:
                logger.exception(
                    "RAG search failed."
                )

                return {
                    "control_action": "tool_error",
                    "final_reply": TOOL_ERROR_REPLY,
                    "plan_step_index": index + 1,
                }

            if result[
                "source_count"
            ] == 0:
                return {
                    "control_action": "no_sources",
                    "final_reply": NO_SOURCE_REPLY,
                    "plan_step_index": index + 1,
                    "tool_call_count": 1,
                    "tool_evidence": [
                        {
                            "name": "search_knowledge_base",
                            "succeeded": False,
                            "source_count": 0,
                        }
                    ],
                    "trace_events": [
                        {
                            "type": "tool_result",
                            "name": "search_knowledge_base",
                            "input": result[
                                "query"
                            ],
                            "output": result[
                                "content"
                            ],
                        }
                    ],
                }

            return {
                "plan_step_index": index + 1,
                "rag_documents": result[
                    "documents"
                ],
                "rag_context": result[
                    "content"
                ],
                "tool_call_count": 1,
                "tool_evidence": [
                    {
                        "name": "search_knowledge_base",
                        "succeeded": True,
                        "source_count": result[
                            "source_count"
                        ],
                    }
                ],
                "trace_events": [
                    {
                        "type": "tool_result",
                        "name": "search_knowledge_base",
                        "input": result[
                            "query"
                        ],
                        "output": result[
                            "content"
                        ],
                    }
                ],
            }

        # --------------------------------------------------------------------
        # CALCULATE
        # --------------------------------------------------------------------

        if step["type"] == "calculate":

            expression = step.get(
                "expression",
                "",
            )

            if not expression.strip():

                try:
                    expression = (
                        await self._build_calculation_expression(
                            state
                        )
                    )
                except Exception:
                    logger.exception(
                        "Could not build calculation expression."
                    )

                    return {
                        "control_action": "calculation_error",
                        "final_reply": CALCULATION_REQUIRED_REPLY,
                        "plan_step_index": index + 1,
                    }

            if not expression.strip():
                return {
                    "control_action": "calculation_error",
                    "final_reply": CALCULATION_REQUIRED_REPLY,
                    "plan_step_index": index + 1,
                }

            try:
                result = await self._execute_calculate(
                    expression
                )
            except Exception:
                logger.exception(
                    "calculate failed."
                )

                return {
                    "control_action": "calculation_error",
                    "final_reply": CALCULATION_REQUIRED_REPLY,
                    "plan_step_index": index + 1,
                }

            return {
                "plan_step_index": index + 1,
                "calculation_expression": expression,
                "calculation_result": result,
                "tool_call_count": 1,
                "tool_evidence": [
                    {
                        "name": "calculate",
                        "succeeded": True,
                        "source_count": 0,
                    }
                ],
                "trace_events": [
                    {
                        "type": "tool_result",
                        "name": "calculate",
                        "input": expression,
                        "output": result,
                    }
                ],
            }

        # --------------------------------------------------------------------
        # ANSWER
        # --------------------------------------------------------------------

        if step["type"] == "answer":
            return {
                "plan_step_index": index + 1,
            }

        return {
            "plan_step_index": index + 1,
        }

    # ========================================================================
    # RAG
    # ========================================================================

    async def _execute_search(
        self,
        step: dict[str, Any],
    ) -> dict[str, Any]:

        query = step[
            "query"
        ]

        started = perf_counter()

        with trace_span(
            "tool.search_knowledge_base",
            {
                "openinference.span.kind": "TOOL",
                "tool.name": "search_knowledge_base",
            },
        ) as span:

            record_tool_input(
                span,
                query,
            )

            documents = await self._retriever.search(
                query
            )

            span.set_attribute(
                "retrieval.documents",
                len(documents),
            )

        sourced_documents = [
            document
            for document in documents
            if (
                isinstance(
                    document,
                    dict,
                )
                and isinstance(
                    document.get(
                        "metadata"
                    ),
                    dict,
                )
                and any(
                    document[
                        "metadata"
                    ].get(field)
                    for field in (
                        "source",
                        "file_name",
                        "dok_id",
                    )
                )
            )
        ]

        # --------------------------------------------------------------------
        # Keep the retriever's order, but do not require the model to use
        # the first result.
        # --------------------------------------------------------------------

        relevant_documents = (
            sourced_documents
        )

        chunks: list[str] = []

        for index, document in enumerate(
            relevant_documents
        ):
            metadata = document[
                "metadata"
            ]

            dok_id = metadata.get(
                "dok_id",
                "Quelle",
            )

            title = metadata.get(
                "title",
                "Dokument",
            )

            keywords = document.get(
                "keywords",
                [],
            )

            if not isinstance(
                keywords,
                list,
            ):
                keywords = []

            chunks.append(
                load_prompt(
                    "rag/rag_source_context.txt",
                    index=str(index + 1),
                    dok_id=str(dok_id),
                    title=str(title),
                    file_name=str(metadata.get("file_name", "")),
                    retrieval_mode=str(
                        document.get("retrieval_mode", "")
                    ),
                    summary=str(document.get("summary", "")),
                    keywords=", ".join(map(str, keywords)),
                    content=str(document.get("content", "")),
                )
            )

        if chunks:
            content = "\n\n".join(
                chunks
            )
        else:
            content = load_prompt("rag/rag_no_results.txt")

        record_tool_output(
            span,
            content,
        )

        logger.info(
            "RAG search completed: sources=%d, %.0f ms",
            len(relevant_documents),
            (
                perf_counter()
                - started
            )
            * 1000,
        )

        return {
            "query": query,
            "documents": relevant_documents,
            "content": content,
            "source_count": len(
                relevant_documents
            ),
        }

    # ========================================================================
    # Calculation expression
    # ========================================================================

    async def _build_calculation_expression(
        self,
        state: AgentState,
    ) -> str:

        user_question = _latest_user_question(
            state
        )

        rag_context = state.get(
            "rag_context",
            "",
        )

        prompt = load_prompt(
            "tools/calculation_expression.txt",
            user_question=user_question,
            rag_context=rag_context,
        )

        response_schema = {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
            "additionalProperties": False,
        }
        last_error: ValueError | None = None
        for attempt in range(2):
            messages = [{"role": "system", "content": prompt}]
            if attempt:
                messages.append(
                    {
                        "role": "user",
                        "content": load_prompt(
                            "tools/calculation_expression_retry.txt"
                        ),
                    }
                )
            response = await self._client.chat_completion(
                messages=messages,
                max_tokens=self._config.calculation_max_tokens,
                temperature=self._config.agent_temperature,
                response_format=response_schema,
                think=False,
                stream=False,
            )
            if not response.choices:
                last_error = ValueError(
                    "Kein Berechnungsausdruck erzeugt."
                )
                continue

            content = _value(
                response.choices[0].message,
                "content",
                "",
            )
            done_reason = _value(response, "done_reason")
            try:
                if not isinstance(content, str) or done_reason == "length":
                    raise ValueError(
                        "Das Modell lieferte keine vollständige JSON-Antwort."
                    )
                parsed = json.loads(content)
                expression = (
                    parsed.get("expression")
                    if isinstance(parsed, dict)
                    else None
                )
                if not isinstance(expression, str) or not expression.strip():
                    raise ValueError("Das JSON enthält keinen Rechenausdruck.")
                expression = expression.strip()
                if not re.fullmatch(r"[0-9\s.,()+*/%\-]+", expression):
                    raise ValueError("Der Rechenausdruck enthält ungültige Zeichen.")
                calculate(expression)
            except (json.JSONDecodeError, ValueError) as error:
                last_error = ValueError(
                    "Das Modell lieferte keinen gültigen ausführbaren "
                    "Rechenausdruck."
                )
                logger.warning(
                    "Calculation expression response rejected (attempt %d/2: %s).",
                    attempt + 1,
                    type(error).__name__,
                )
                continue

            logger.info("Generated a valid calculation expression.")
            return expression

        raise last_error or ValueError("Kein Berechnungsausdruck erzeugt.")

    # ========================================================================
    # Calculate
    # ========================================================================

    async def _execute_calculate(
        self,
        expression: str,
    ) -> str:

        started = perf_counter()

        with trace_span(
            "tool.calculate",
            {
                "openinference.span.kind": "TOOL",
                "tool.name": "calculate",
            },
        ) as span:

            record_tool_input(
                span,
                expression,
            )

            result = calculate(
                expression
            )

            record_tool_output(
                span,
                result,
            )

        logger.info(
            "calculate completed: %.0f ms",
            (
                perf_counter()
                - started
            )
            * 1000,
        )

        return str(
            result
        )

    # ========================================================================
    # Finalization
    # ========================================================================

    async def _finalize(
        self,
        state: AgentState,
    ) -> dict[str, Any]:

        control_action = state.get(
            "control_action"
        )

        if control_action == "no_sources":
            reply = NO_SOURCE_REPLY

        elif control_action in {
            "planning_error",
            "calculation_error",
            "tool_error",
        }:
            reply = (
                state.get(
                    "final_reply"
                )
                or TOOL_ERROR_REPLY
            )

        else:
            try:
                reply = await self._generate_final_answer(
                    state
                )
            except Exception:
                logger.exception(
                    "Final answer generation failed."
                )
                reply = TOOL_ERROR_REPLY

        return {
            "final_reply": reply,
            "trace_events": [
                {
                    "type": "final",
                    "reply": reply,
                }
            ],
        }

    # ========================================================================
    # Final answer generation
    # ========================================================================

    async def _generate_final_answer(
        self,
        state: AgentState,
    ) -> str:

        user_question = _latest_user_question(
            state
        )

        rag_context = state.get(
            "rag_context",
            "",
        )

        calculation_result = state.get(
            "calculation_result",
            "",
        )

        calculation_expression = state.get(
            "calculation_expression",
            "",
        )

        prompt = load_prompt(
            "agent/final_answer.txt",
            user_question=user_question,
            rag_context=rag_context,
            calculation_expression=calculation_expression,
            calculation_result=calculation_result,
        )

        response_schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "maxLength": self._config.final_answer_max_chars,
                },
            },
            "required": ["answer"],
            "additionalProperties": False,
        }
        response = await self._client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT
                        + "\n\n"
                        + RAG_SYSTEM_PROMPT
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=self._config.final_answer_max_tokens,
            temperature=self._config.agent_temperature,
            response_format=response_schema,
            think=False,
            stream=False,
        )

        if not response.choices:
            return TOOL_ERROR_REPLY

        content = _value(
            response.choices[0].message,
            "content",
            "",
        )
        done_reason = _value(response, "done_reason")

        if not isinstance(content, str) or done_reason == "length":
            logger.warning("Final answer was missing or truncated.")
            return TOOL_ERROR_REPLY

        try:
            parsed_answer = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Final answer was not valid structured JSON.")
            return TOOL_ERROR_REPLY

        answer = (
            parsed_answer.get("answer")
            if isinstance(parsed_answer, dict)
            else None
        )

        if not isinstance(answer, str) or not answer.strip():
            logger.warning("Final answer JSON did not contain a usable answer.")
            return TOOL_ERROR_REPLY

        answer = answer.strip()

        # --------------------------------------------------------------------
        # Letztes Safety-Net: list retrieved sources without implying they were
        # all used or preferring the first result.
        # --------------------------------------------------------------------

        documents = state.get(
            "rag_documents",
            [],
        )

        if documents:
            references = []
            for document in documents:
                metadata = document.get("metadata", {})
                if not isinstance(metadata, dict):
                    continue
                dok_id = metadata.get("dok_id")
                title = metadata.get(
                    "title",
                    metadata.get("file_name"),
                )
                if dok_id and title:
                    reference = f"[{dok_id}: {title}]"
                    if reference not in references:
                        references.append(reference)

            if not re.search(
                r"\[[^\]]+:\s*[^\]]+\]",
                answer,
            ) and references:
                answer = (
                    f"{answer}\n\n"
                    + load_prompt(
                        "responses/found_sources_fallback.txt",
                        references="\n".join(
                            f"- {reference}" for reference in references
                        ),
                    )
                )

        return answer
