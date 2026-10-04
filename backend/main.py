import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from backend.agent import ChatAgent
from backend.local_llm import LocalModelError
from backend.session_logging import SessionEventLogger

load_dotenv(Path(__file__).with_name(".env"))

logger = logging.getLogger(__name__)
session_logger = SessionEventLogger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.chatbot = None
    try:
        app.state.chatbot = ChatAgent(
            model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3:4b"),
            embedding_model=os.getenv(
                "OLLAMA_EMBEDDING_MODEL",
                "nomic-embed-text",
            ),
            keyword_model=os.getenv("OLLAMA_KEYWORD_MODEL", "qwen3:4b"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            documents_directory=Path(__file__).resolve().parent.parent / "anlagen",
        )
        await app.state.chatbot.initialize()
        yield
    finally:
        if app.state.chatbot is not None:
            await app.state.chatbot.close()


app = FastAPI(title="Demo Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[ConversationTurn] = Field(default_factory=list, max_length=20)
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
                if event.get("type") in {"answer_delta", "final", "error"}:
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
