import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError
from starlette.requests import Request

import backend.main as chat_api
from backend.main import ChatRequest
from backend.session_logging import SessionEventLogger


class FakeStreamingAgent:
    async def stream(self, message, history=None):
        yield {"type": "answer_delta", "text": "Das Ergebnis ist "}
        yield {"type": "llm_request", "messages": [{"content": message}]}
        yield {
            "type": "llm_response",
            "thinking": "Interne Überlegung",
            "content": "",
            "tool_calls": [{"name": "calculate"}],
        }
        yield {"type": "tool_start", "name": "calculate"}
        yield {"type": "tool_result", "name": "calculate", "output": "30"}
        yield {"type": "final", "reply": "Das Ergebnis ist 30."}


class ChatRequestTests(unittest.TestCase):
    def test_history_accepts_normal_long_assistant_responses(self):
        request = ChatRequest.model_validate(
            {
                "message": "Zuschuss",
                "history": [
                    {"role": "user", "content": "Zahnreinigung"},
                    {
                        "role": "assistant",
                        "content": "Antwort " * 250,
                    },
                ],
            }
        )

        self.assertEqual(len(request.history), 2)

    def test_history_rejects_turns_exceeding_context_limit(self):
        with self.assertRaises(ValidationError):
            ChatRequest.model_validate(
                {
                    "message": "Zuschuss",
                    "history": [
                        {"role": "assistant", "content": "A" * 4001},
                    ],
                }
            )

    def test_chat_stream_sends_only_final_answer_and_logs_session_events(self):
        session_id = UUID("01234567-89ab-cdef-0123-456789abcdef")
        request_data = ChatRequest(
            message="Was sind 15 % von 200?",
            session_id=session_id,
        )
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/chat",
            "raw_path": b"/api/chat",
            "query_string": b"",
            "headers": [],
            "server": ("testserver", 80),
            "client": ("testclient", 12345),
            "app": chat_api.app,
        }
        old_chatbot = getattr(chat_api.app.state, "chatbot", None)
        chat_api.app.state.chatbot = FakeStreamingAgent()

        with tempfile.TemporaryDirectory() as directory:
            old_logger = chat_api.session_logger
            chat_api.session_logger = SessionEventLogger(Path(directory))
            try:
                async def invoke():
                    response = await chat_api.chat(request_data, Request(scope))
                    chunks = [
                        chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                        async for chunk in response.body_iterator
                    ]
                    return "".join(chunks)

                stream = asyncio.run(invoke())
                streamed_events = [
                    json.loads(line[6:])
                    for line in stream.splitlines()
                    if line.startswith("data: ")
                ]
                log_paths = list(
                    Path(directory).glob(f"session-*-{session_id}.jsonl")
                )
                self.assertEqual(len(log_paths), 1)
                log_path = log_paths[0]
                logged_events = [
                    json.loads(line)
                    for line in log_path.read_text(encoding="utf-8").splitlines()
                ]
            finally:
                chat_api.session_logger = old_logger
                chat_api.app.state.chatbot = old_chatbot

        self.assertEqual(
            [event["type"] for event in streamed_events],
            ["status", "answer_delta", "final"],
        )
        self.assertEqual(
            [event["event_type"] for event in logged_events],
            [
                "user_input",
                "llm_request",
                "llm_response",
                "tool_start",
                "tool_result",
                "final",
            ],
        )
        self.assertEqual(
            logged_events[2]["payload"]["thinking"],
            "Interne Überlegung",
        )


if __name__ == "__main__":
    unittest.main()
