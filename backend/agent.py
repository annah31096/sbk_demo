"""LangGraph agent backed by local Ollama models."""

import asyncio
import json
import logging
import operator
import re
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.local_llm import LocalOllamaClient
from backend.rag import MarkdownRetriever
from backend.tools import calculate

PROMPTS_DIR = Path(__file__).with_name("prompts")
SYSTEM_PROMPT = (PROMPTS_DIR / "system.txt").read_text(encoding="utf-8").strip()
TOOL_SELECTION_PROMPT = (
    PROMPTS_DIR / "tool_selection.txt"
).read_text(encoding="utf-8").strip()
CALCULATE_TOOL_PROMPT = (
    PROMPTS_DIR / "calculate_tool.txt"
).read_text(encoding="utf-8").strip()
CALCULATE_TOOL_INPUT_PROMPT = (
    PROMPTS_DIR / "calculate_tool_input.txt"
).read_text(encoding="utf-8").strip()
RAG_SYSTEM_PROMPT = (
    PROMPTS_DIR / "rag_system.txt"
).read_text(encoding="utf-8").strip()
SEARCH_KNOWLEDGE_TOOL_PROMPT = (
    PROMPTS_DIR / "search_knowledge_tool.txt"
).read_text(encoding="utf-8").strip()
SEARCH_KNOWLEDGE_INPUT_PROMPT = (
    PROMPTS_DIR / "search_knowledge_input.txt"
).read_text(encoding="utf-8").strip()

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
NUMBER = r"\d+(?:[.,]\d+)?"
PERCENT_CALCULATION = re.compile(
    rf"{NUMBER}\s*(?:%|prozent)\s*(?:von|of|[*x×])\s*{NUMBER}"
    rf"|{NUMBER}\s+(?:ist|sind)\s+wie viel\s*(?:%|prozent)\s*"
    rf"(?:von)?\s*{NUMBER}",
    re.IGNORECASE,
)
CALCULATION_REQUEST = re.compile(
    r"\b(?:berechne|berechnen|rechne|ausrechnen|ermittle|ermitteln)\b",
    re.IGNORECASE,
)
PERCENTAGE_VALUE = re.compile(
    rf"(?P<percentage>{NUMBER})\s*(?:%|\bprozent\b)",
    re.IGNORECASE,
)
CALCULATION_AMOUNT = re.compile(
    r"(?<![\w/])(?P<amount>\d+(?:[.,]\d{1,2})?)\s*(?:€|euro)?",
    re.IGNORECASE,
)
GKV_QUESTION = re.compile(
    r"\b(?:"
    r"gesetzlich(?:e[nrms]?)?\s+krank(?:en)?(?:kasse|versicherung)"
    r"|krankenkasse(?:n)?|krankversicherung|krankenversicherung"
    r"|\b\w*versicherung\w*\b"
    r"|musterkasse|gkv|krankengeld|zuzahlung(?:en)?|belastungsgrenze"
    r"|familienversicherung|heilmittel|hilfsmittel|heilpraktiker|kasse(?:n)?"
    r"|häusliche\s+krankenpflege|haushaltshilfe|rezept|e[- ]?rezept"
    r"|zahnersatz|zahnreinigung|bonusprogramm|krankenhausaufenthalt"
    r"|krankenkassenbeitrag|versichert(?:e[nr]?)?"
    r"|krank(?:heit|en|er|e)?\b|arzt(?:besuch|termin|kosten)?|ärzt(?:lich|e[nr]?)"
    r"|beitrag(?:s)?|pflegeversicherung|krankheitsfall|gesundheit"
    r"|vorsorge(?:untersuchung)?|schutzimpfung|impfungen?"
    r"|rehabilitation|rehakur|psychotherapie|physiotherapie"
    r"|ergotherapie|logopädie|podologie|fahrkosten|zweitmeinung"
    r"|videosprechstunde|kinderuntersuchungen?|widerspruch"
    r"|diga|digitale\s+gesundheitsanwendungen"
    r"|elektronische\s+patientenakte|auslandsreisen?"
    r")\b",
    re.IGNORECASE,
)
SPECIFIC_QUESTION = re.compile(
    r"\b(?:"
    r"wie|was|wer|wann|wo|warum|wieso|welche[nrms]?|wieviel|wie\s+viel"
    r"|kost(?:et|en)|zahlt|übernimmt|erstattet|erstattung|bekomme|anspruch"
    r"|voraussetzung|zuzahlung|kosten|betrag|höhe|beantragen|abrechnen|gilt|gelten|darf|kann"
    r"|muss|müssen|hilfe|zuschuss|leistung|leistungen"
    r")\b",
    re.IGNORECASE,
)
CONTEXTUAL_FOLLOW_UP = re.compile(
    r"^\s*(?:"
    r"und\b|dazu\b|dafür\b|davon\b|damit\b|"
    r"wie\s+viel\b|was\s+ist\s+mit\b|"
    r"bei\s+(?:kindern?|mir|ihm|ihr|uns|ihnen)\b|"
    r"das\b|diese[rsm]?\b|dort\b|"
    r"voraussetzungen?\b|kosten\b|zuzahlung\b|zuschuss\b|"
    r"erstattung\b|einkommen\b|regelfall\b|chronisch\b|"
    r"betrag\b|berechnung\b"
    r")",
    re.IGNORECASE,
)
CONTEXTUAL_CALCULATION = re.compile(
    r"\b(?:berechn?(?:e|en)?|rechne|ausrechnen|berechnung)\b.*"
    r"\b(?:einkommen|betrag|euro|€|monatlich|jährlich|prozent|%)\b",
    re.IGNORECASE,
)
CLARIFICATION_QUESTION = "Welche konkrete Frage hast du dazu?"
NO_SOURCE_REPLY = (
    "Ich finde in den bereitgestellten Anlagen keine passende Quelle zu deiner "
    "Frage. Damit ich nichts erfinde, kann ich dazu keine verlässliche Antwort "
    "geben. Kannst du die Frage eingrenzen oder sagen, um welche Leistung oder "
    "Regelung es geht?"
)
OUT_OF_SCOPE_REPLY = (
    "Ich kann nur Fragen zu den zugelassenen Themen der gesetzlichen "
    "Krankenversicherung anhand der bereitgestellten Quellen beantworten. "
    "Bitte stelle eine passende Frage."
)
TOOL_FAILURE_REPLY = (
    "Ich konnte die benötigte Prüfung gerade nicht zuverlässig abschließen. "
    "Deshalb gebe ich keine Sachantwort, die möglicherweise nicht belegt ist. "
    "Bitte versuche es erneut."
)
TOOL_ERROR_REPLY = (
    "Für diese Frage konnte ich keine verlässliche Antwort aus den "
    "bereitgestellten Quellen ermitteln. Bitte versuche es erneut oder "
    "formuliere die Frage genauer."
)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[dict[str, Any]], operator.add]
    tool_call_count: Annotated[int, operator.add]
    trace_events: Annotated[list[dict[str, Any]], operator.add]
    final_reply: str
    control_action: str
    model_done_reason: str
    on_answer_token: Any
    tool_evidence: Annotated[list[dict[str, Any]], operator.add]


def _message_value(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _tool_call_value(tool_call: Any, key: str, default: Any = None) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get(key, default)
    return getattr(tool_call, key, default)


def _function_value(tool_call: Any, key: str, default: Any = None) -> Any:
    function = _tool_call_value(tool_call, "function", {})
    return _message_value(function, key, default)


class ChatAgent:
    def __init__(
        self,
        model: str,
        embedding_model: str = "nomic-embed-text",
        keyword_model: str | None = None,
        ollama_host: str = "http://127.0.0.1:11434",
        documents_directory: Path | None = None,
        retriever: MarkdownRetriever | None = None,
        chat_client: Any | None = None,
    ) -> None:
        self._client = chat_client or LocalOllamaClient(ollama_host, model)
        self._model = model
        self._retriever = retriever if retriever is not None else MarkdownRetriever(
            documents_directory
            or Path(__file__).resolve().parent.parent / "anlagen",
            embedding_model,
            keyword_model=keyword_model or model,
            ollama_host=ollama_host,
        )
        graph = StateGraph(AgentState)
        graph.add_node("route_request", self._route_request)
        graph.add_node("prepare_model_call", self._prepare_model_call)
        graph.add_node("call_model", self._call_model)
        graph.add_node("record_model_response", self._record_model_response)
        graph.add_node("announce_tool", self._announce_tool)
        graph.add_node("run_tools", self._run_tools)
        graph.add_node("validate_final", self._validate_final)
        graph.add_node("finalize_guarded_reply", self._finalize_guarded_reply)
        graph.add_conditional_edges(
            "route_request",
            self._route_destination,
        )
        graph.add_edge("prepare_model_call", "call_model")
        graph.add_edge("call_model", "record_model_response")
        graph.add_conditional_edges(
            "record_model_response",
            lambda state: (
                "announce_tool"
                if state["messages"][-1].get("tool_calls")
                else "validate_final"
            ),
        )
        graph.add_conditional_edges(
            "validate_final",
            lambda state: (
                "finalize_guarded_reply"
                if state["control_action"]
                else END
            ),
        )
        graph.add_edge("announce_tool", "run_tools")
        graph.add_conditional_edges(
            "run_tools",
            lambda state: (
                "finalize_guarded_reply"
                if state["control_action"]
                else "prepare_model_call"
            ),
        )
        graph.add_edge("finalize_guarded_reply", END)
        graph.add_edge(START, "route_request")
        self._graph = graph.compile()

    async def initialize(self) -> None:
        await self._retriever.initialize()

    async def stream(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        event_queue: asyncio.Queue[Any] = asyncio.Queue()

        def on_answer_token(token: str) -> None:
            event_queue.put_nowait({"type": "answer_delta", "text": token})

        conversation = [
            {"role": item["role"], "content": item["content"]}
            for item in (history or [])
            if item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
        ]
        conversation.append({"role": "user", "content": message})
        initial_state: AgentState = {
            "messages": conversation,
            "tool_call_count": 0,
            "trace_events": [],
            "final_reply": "",
            "control_action": "",
            "model_done_reason": "",
            "on_answer_token": on_answer_token,
            "tool_evidence": [],
        }

        async def run_graph() -> None:
            try:
                async for event in self._graph.astream(
                    initial_state,
                    stream_mode="updates",
                    version="v2",
                ):
                    if not isinstance(event, dict) or event.get("type") != "updates":
                        continue
                    updates = event.get("data")
                    if not isinstance(updates, dict):
                        continue
                    for node_update in updates.values():
                        if not isinstance(node_update, dict):
                            continue
                        for trace_event in node_update.get("trace_events", []):
                            if isinstance(trace_event, dict):
                                event_queue.put_nowait(trace_event)
            except Exception as error:
                event_queue.put_nowait(error)
            finally:
                event_queue.put_nowait(None)

        graph_task = asyncio.create_task(run_graph())
        try:
            while True:
                event = await event_queue.get()
                if event is None:
                    break
                if isinstance(event, Exception):
                    raise event
                if isinstance(event, dict):
                    yield event
            await graph_task
        finally:
            if not graph_task.done():
                graph_task.cancel()
            await asyncio.gather(graph_task, return_exceptions=True)

    async def close(self) -> None:
        await self._client.close()
        await self._retriever.close()

    def _route_request(self, state: AgentState) -> dict[str, Any]:
        message = state["messages"][-1]["content"]
        calculation_expression = self._percentage_calculation_expression(
            state["messages"]
        )
        if calculation_expression:
            tool_name = "calculate"
            arguments = {"expression": calculation_expression}
            tool_id = "percentage-calculation"
        elif not PERCENT_CALCULATION.search(message):
            search_query = self._contextual_query(state["messages"])
            prior_gkv_context = any(
                item.get("role") == "user"
                and isinstance(item.get("content"), str)
                and GKV_QUESTION.search(item["content"])
                for item in state["messages"][:-1]
            )
            is_contextual_follow_up = (
                prior_gkv_context
                and (
                    CONTEXTUAL_FOLLOW_UP.search(message)
                    or CONTEXTUAL_CALCULATION.search(message)
                )
            )
            if not GKV_QUESTION.search(message) and not is_contextual_follow_up:
                return {
                    "control_action": "out_of_scope",
                    "final_reply": OUT_OF_SCOPE_REPLY,
                }
            needs_clarification = (
                len(state["messages"]) == 1
                and not SPECIFIC_QUESTION.search(message)
            )
            tool_name = "search_knowledge_base"
            arguments = {"query": search_query}
            tool_id = (
                "clarification-suggestions"
                if needs_clarification
                else "health-insurance-knowledge-search"
            )
        else:
            tool_name = "calculate"
            arguments = {"expression": message}
            tool_id = "percentage-calculation"

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(
                                    arguments,
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                }
            ]
        }

    @staticmethod
    def _percentage_calculation_expression(
        messages: list[dict[str, Any]],
    ) -> str | None:
        message = messages[-1]["content"]
        if not CALCULATION_REQUEST.search(message):
            return None

        context = "\n".join(
            item.get("content", "")
            for item in messages
            if item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
        )
        percentages = list(PERCENTAGE_VALUE.finditer(context))
        amounts = list(CALCULATION_AMOUNT.finditer(message))
        if not percentages or not amounts:
            return None

        percentage = percentages[-1].group("percentage").replace(",", ".")
        amount = amounts[-1].group("amount").replace(",", ".")
        annual_percentage = bool(
            re.search(r"\b(?:jährlich|jährliche|pro\s+jahr|kalenderjahr)\b", context, re.I)
        )
        monthly_amount = bool(
            re.search(r"\b(?:monatlich|pro\s+monat|im\s+monat)\b", message, re.I)
        )
        if annual_percentage and monthly_amount:
            return f"{amount} * 12 * {percentage} %"
        return f"{amount} * {percentage} %"

    @staticmethod
    def _route_destination(state: AgentState) -> str:
        if state["control_action"]:
            return "finalize_guarded_reply"
        if state["messages"][-1].get("tool_calls"):
            return "announce_tool"
        return "prepare_model_call"

    @staticmethod
    def _contextual_query(messages: list[dict[str, Any]]) -> str:
        current_message = messages[-1]["content"]
        user_context = [
            item["content"]
            for item in messages[:-1]
            if item.get("role") == "user"
            and isinstance(item.get("content"), str)
        ]
        return " ".join(
            part.strip()
            for part in [*user_context, current_message]
            if part.strip()
        )

    def _prepare_model_call(self, state: AgentState) -> dict[str, Any]:
        return {
            "trace_events": [
                {
                    "type": "llm_request",
                    "model": self._model,
                    "messages": self._model_messages(state),
                    "tools": [CALCULATE_TOOL, SEARCH_KNOWLEDGE_TOOL],
                }
            ]
        }

    def _model_messages(self, state: AgentState) -> list[dict[str, Any]]:
        system_prompt = SYSTEM_PROMPT
        if any(
            message.get("role") == "tool"
            and message.get("name") == "search_knowledge_base"
            for message in state["messages"]
        ):
            system_prompt = f"{system_prompt}\n\n{RAG_SYSTEM_PROMPT}"
        return [
            {
                "role": "system",
                "content": f"{system_prompt}\n\n{TOOL_SELECTION_PROMPT}",
            },
            *state["messages"],
        ]

    async def _call_model(self, state: AgentState) -> dict[str, Any]:
        response = await self._client.chat_completion(
            messages=self._model_messages(state),
            tools=[CALCULATE_TOOL, SEARCH_KNOWLEDGE_TOOL],
            max_tokens=4096,
            think=True,
            stream=True,
            on_token=state["on_answer_token"],
        )
        if not response.choices:
            raise RuntimeError("Das lokale Chat-Modell lieferte keine Antwort.")
        response_message = response.choices[0].message
        content = _message_value(response_message, "content")
        thinking = _message_value(response_message, "thinking", "")
        done_reason = _message_value(response, "done_reason")
        if content is not None and not isinstance(content, str):
            raise RuntimeError("Das lokale Chat-Modell lieferte ein ungültiges Format.")
        if not isinstance(thinking, str):
            raise RuntimeError("Das lokale Chat-Modell lieferte ungültige Thinking-Daten.")
        raw_tool_calls = _message_value(response_message, "tool_calls") or []
        if raw_tool_calls and state["tool_call_count"] >= 3:
            raise RuntimeError("Der Agent hat das Limit für Werkzeugaufrufe erreicht.")
        tool_calls = []
        for tool_call in raw_tool_calls:
            function_name = _function_value(tool_call, "name", "")
            if function_name not in {"calculate", "search_knowledge_base"}:
                logger.warning("Ignoring unsupported agent tool: %s", function_name)
                continue
            tool_calls.append(
                {
                    "id": _tool_call_value(tool_call, "id", "calculate"),
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": _function_value(tool_call, "arguments", "{}"),
                    },
                }
            )

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content or "",
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        return {
            "messages": [assistant_message],
            "model_done_reason": done_reason or "",
            "trace_events": [
                {
                    "type": "llm_response",
                    "content": assistant_message["content"],
                    "thinking": thinking,
                    "done_reason": done_reason,
                    "tool_calls": tool_calls,
                }
            ],
        }

    def _record_model_response(self, state: AgentState) -> dict[str, Any]:
        assistant_message = state["messages"][-1]
        if state["model_done_reason"] == "length":
            raise RuntimeError(
                "Das Chat-Modell hat sein Ausgabelimit erreicht; die Antwort "
                "wurde nicht vollständig erzeugt und deshalb nicht angezeigt."
            )
        content = assistant_message["content"]
        tool_calls = assistant_message.get("tool_calls", [])
        if not tool_calls and not content.strip():
            raise RuntimeError("Das lokale Chat-Modell lieferte eine leere Antwort.")
        if not tool_calls:
            return {
                "final_reply": content,
            }
        return {}

    def _announce_tool(self, state: AgentState) -> dict[str, Any]:
        trace_events = []
        for tool_call in state["messages"][-1].get("tool_calls", []):
            tool_name = _function_value(tool_call, "name", "")
            if tool_name not in {"calculate", "search_knowledge_base"}:
                continue
            tool_id = _tool_call_value(tool_call, "id", "calculate")
            trace_events.append(
                {
                    "type": "tool_start",
                    "name": tool_name,
                    "input": _function_value(tool_call, "arguments", {}),
                    "reason": (
                        "Prozentrechnung erkannt"
                        if tool_id == "percentage-calculation"
                        else "RAG-Suche für Rückfrage und Themenvorschläge"
                        if tool_id == "clarification-suggestions"
                        else "Frage zur gesetzlichen Krankenversicherung erkannt"
                        if tool_id == "health-insurance-knowledge-search"
                        else "Vom Modell angefordert"
                    ),
                }
            )
        return {"trace_events": trace_events}

    async def _run_tools(self, state: AgentState) -> dict[str, Any]:
        tool_results = []
        trace_events = []
        tool_evidence = []
        control_action = ""
        final_reply = ""
        for tool_call in state["messages"][-1].get("tool_calls", []):
            tool_name = _function_value(tool_call, "name", "")
            if tool_name not in {"calculate", "search_knowledge_base"}:
                continue

            tool_id = _tool_call_value(tool_call, "id", "calculate")
            expression = ""
            query = ""
            source_count = 0
            tool_succeeded = False
            raw_arguments = _function_value(tool_call, "arguments", {})
            try:
                arguments = raw_arguments
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Die Werkzeugargumente sind ungültig.")
                if tool_name == "calculate":
                    expression = arguments.get("expression")
                    if not isinstance(expression, str) or not expression.strip():
                        raise ValueError("Es wurde kein Rechenausdruck übergeben.")
                    result = calculate(expression)
                    tool_succeeded = True
                else:
                    query = arguments.get("query")
                    if not isinstance(query, str) or not query.strip():
                        raise ValueError("Es wurde keine Suchfrage übergeben.")
                    documents = await self._retriever.search(query)
                    sourced_documents = [
                        document
                        for document in documents
                        if isinstance(document, dict)
                        and isinstance(document.get("metadata"), dict)
                        and any(
                            document["metadata"].get(field)
                            for field in ("source", "file_name", "dok_id")
                        )
                    ]
                    source_count = len(sourced_documents)
                    if source_count:
                        result = "\n\n".join(
                            (
                                f"Suchmodus: {document['retrieval_mode']}\n"
                                f"[{document['metadata'].get('dok_id', 'Quelle')}: "
                                f"{document['metadata'].get('title', 'Dokument')}] "
                                f"(Datei: {document['metadata']['file_name']})\n"
                                f"Zusammenfassung: {document['summary']}\n"
                                f"Keywords: {', '.join(document['keywords'])}\n"
                                f"{document['content']}"
                            )
                            for document in sourced_documents
                        )
                        tool_succeeded = True
                        if tool_id == "clarification-suggestions":
                            control_action = "clarification"
                            final_reply = self._clarification_reply(
                                sourced_documents
                            )
                    else:
                        result = (
                            "In den Anlagen wurden keine passenden Informationen "
                            "zu dieser Frage gefunden."
                        )
                        control_action = "no_sources"
                        final_reply = NO_SOURCE_REPLY
            except (KeyError, TypeError, ValueError, RuntimeError, OSError):
                logger.exception("Agent tool %s failed.", tool_name)
                result = "Tool-Fehler: Die Werkzeugausführung ist fehlgeschlagen."
                control_action = "tool_error"
                final_reply = TOOL_ERROR_REPLY

            tool_evidence.append(
                {
                    "name": tool_name,
                    "succeeded": tool_succeeded,
                    "source_count": source_count,
                }
            )

            trace_events.append(
                {
                    "type": "tool_result",
                    "name": tool_name,
                    "input": expression if tool_name == "calculate" else query,
                    "output": result,
                }
            )
            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": result,
                }
            )
        return {
            "messages": tool_results,
            "tool_call_count": len(tool_results),
            "tool_evidence": tool_evidence,
            "trace_events": trace_events,
            "control_action": control_action,
            "final_reply": final_reply,
        }

    def _validate_final(self, state: AgentState) -> dict[str, Any]:
        evidence = state["tool_evidence"]
        if not evidence:
            return {
                "control_action": "missing_tool",
                "final_reply": TOOL_FAILURE_REPLY,
            }
        if any(
            item["name"] == "search_knowledge_base"
            and item["source_count"] == 0
            for item in evidence
        ):
            return {
                "control_action": "no_sources",
                "final_reply": NO_SOURCE_REPLY,
            }
        if not any(item["succeeded"] for item in evidence):
            return {
                "control_action": "tool_error",
                "final_reply": TOOL_ERROR_REPLY,
            }
        return {
            "trace_events": [
                {"type": "final", "reply": state["final_reply"]}
            ]
        }

    @staticmethod
    def _clarification_reply(
        documents: list[dict[str, Any]],
    ) -> str:
        titles = list(
            dict.fromkeys(
                document["metadata"].get("title", "dieses Thema")
                for document in documents
            )
        )
        topics = ", ".join(f"„{title}“" for title in titles[:3])
        suggestions = (
            f"Zu deinem Thema finde ich passende Informationen, zum Beispiel "
            f"zu {topics}. Geht es dir um Voraussetzungen, Kosten oder eine "
            f"Erstattung? {CLARIFICATION_QUESTION}"
        )
        return suggestions

    def _finalize_guarded_reply(self, state: AgentState) -> dict[str, Any]:
        return {
            "trace_events": [
                {"type": "final", "reply": state["final_reply"]}
            ]
        }
