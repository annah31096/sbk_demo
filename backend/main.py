import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from backend.src.agent import ChatAgent
from backend.config import BackendConfig
from backend.src.llm import LocalModelError
from backend.src.observability.tracing import (
    setup_phoenix_tracing,
    shutdown_phoenix_tracing,
    trace_span,
)
from backend.src.sessions import SessionEventLogger

load_dotenv(Path(__file__).with_name(".env"))
config = BackendConfig.from_env()


def configure_backend_logging() -> None:
    level_name = config.log_level
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        raise ValueError(f"Ungültiges LOG_LEVEL: {level_name}")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger().setLevel(level)
    logging.getLogger("backend").setLevel(level)


configure_backend_logging()
logger = logging.getLogger(__name__)
session_logger = SessionEventLogger(config.session_log_directory)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.chatbot = None
    tracing = setup_phoenix_tracing(
        enabled=config.phoenix_enabled,
        endpoint=config.phoenix_collector_endpoint,
        project_name=config.phoenix_project_name,
        capture_content=config.phoenix_capture_content,
    )
    try:
        app.state.chatbot = ChatAgent(
            config=config,
        )
        await app.state.chatbot.initialize()
        yield
    finally:
        try:
            if app.state.chatbot is not None:
                await app.state.chatbot.close()
        finally:
            shutdown_phoenix_tracing(tracing)


app = FastAPI(title="Demo Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(
        min_length=1,
        max_length=config.max_history_turn_chars,
    )


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=config.max_chat_message_chars)
    history: list[ConversationTurn] = Field(
        default_factory=list,
        max_length=config.max_history_turns,
    )
    session_id: UUID = Field(default_factory=uuid4)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: ChatRequest, http_request: Request) -> StreamingResponse:
    message = request.message.strip()
    chatbot: ChatAgent | None = http_request.app.state.chatbot
    if chatbot is None:
        raise HTTPException(
            status_code=503,
            detail="Der lokale Chatbot ist nicht initialisiert.",
        )

    async def stream_events():
        request_id = uuid4()
        try:
            with trace_span(
                "chat.request",
                {
                    "openinference.span.kind": "CHAIN",
                    "session.id": str(request.session_id),
                    "request.id": str(request_id),
                },
            ):
                session_logger.log(
                    session_id=request.session_id,
                    request_id=request_id,
                    event_type="user_input",
                    payload={"message": message},
                )
                status_event = {
                    "type": "status",
                    "message": "Ich prüfe deine Frage …",
                }
                yield f"data: {json.dumps(status_event, ensure_ascii=False)}\n\n"
                with trace_span(
                    "agent.stream",
                    {"openinference.span.kind": "CHAIN"},
                ):
                    async for event in chatbot.stream(
                        message,
                        history=[
                            turn.model_dump()
                            for turn in request.history
                        ],
                    ):
                        if event.get("type") != "answer_delta":
                            session_logger.log(
                                session_id=request.session_id,
                                request_id=request_id,
                                event_type=event.get("type", "agent_event"),
                                payload=event,
                            )
                        if event.get("type") in {"final", "error"}:
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except (LocalModelError, RuntimeError, ValueError, OSError) as error:
            logger.exception("Agent chat request failed")
            event = {
                "type": "error",
                "message": str(error),
            }
            try:
                session_logger.log(
                    session_id=request.session_id,
                    request_id=request_id,
                    event_type="error",
                    payload=event,
                )
            except OSError:
                logger.exception("Could not persist agent error to session log")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
