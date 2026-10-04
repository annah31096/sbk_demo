import asyncio
import json
import unittest

import httpx

from backend.local_llm import LocalModelError, LocalOllamaClient


class LocalOllamaClientTests(unittest.TestCase):
    def test_chat_tool_call_history_uses_object_arguments(self):
        requests = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"message": {"content": "Antwort"}})

        async def run_request():
            client = httpx.AsyncClient(
                base_url="http://localhost:11434",
                transport=httpx.MockTransport(handler),
            )
            ollama = LocalOllamaClient(
                "http://localhost:11434",
                "qwen3:4b",
                client=client,
            )
            messages = [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_knowledge_base",
                                "arguments": '{"query": "Krankengeld"}',
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "search_knowledge_base",
                    "content": "Informationen zum Krankengeld.",
                },
            ]
            try:
                await ollama.chat_completion(messages=messages)
            finally:
                await ollama.close()
            return messages

        original_messages = asyncio.run(run_request())
        payload = json.loads(requests[0].content)

        self.assertEqual(
            payload["messages"][0]["tool_calls"][0]["function"]["arguments"],
            {"query": "Krankengeld"},
        )
        self.assertEqual(
            original_messages[0]["tool_calls"][0]["function"]["arguments"],
            '{"query": "Krankengeld"}',
        )

    def test_thinking_can_be_enabled_and_is_returned_separately(self):
        requests = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "done_reason": "length",
                    "message": {
                        "thinking": "Interne Überlegung",
                        "content": "Fertige Antwort",
                    }
                },
            )

        async def run_request():
            client = httpx.AsyncClient(
                base_url="http://localhost:11434",
                transport=httpx.MockTransport(handler),
            )
            ollama = LocalOllamaClient(
                "http://localhost:11434",
                "qwen3:4b",
                client=client,
            )
            try:
                return await ollama.chat_completion(
                    messages=[{"role": "user", "content": "Eine Frage"}],
                    think=True,
                )
            finally:
                await ollama.close()

        result = asyncio.run(run_request())
        payload = json.loads(requests[0].content)

        self.assertTrue(payload["think"])
        self.assertEqual(result.choices[0].message.thinking, "Interne Überlegung")
        self.assertEqual(result.choices[0].message.content, "Fertige Antwort")
        self.assertEqual(result.done_reason, "length")

    def test_chat_stream_delivers_only_content_chunks_to_callback(self):
        requests = []
        streamed_response = b"\n".join(
            [
                json.dumps(
                    {
                        "message": {
                            "thinking": "Intern ",
                            "content": "Die Antwort ",
                        },
                        "done": False,
                    }
                ).encode(),
                json.dumps(
                    {
                        "message": {
                            "thinking": "weiter",
                            "content": "ist 30.",
                        },
                        "done": True,
                        "done_reason": "stop",
                    }
                ).encode(),
            ]
        ) + b"\n"

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, content=streamed_response)

        async def run_request():
            client = httpx.AsyncClient(
                base_url="http://localhost:11434",
                transport=httpx.MockTransport(handler),
            )
            ollama = LocalOllamaClient(
                "http://localhost:11434",
                "qwen3:4b",
                client=client,
            )
            content_chunks = []
            try:
                result = await ollama.chat_completion(
                    messages=[{"role": "user", "content": "Rechne"}],
                    stream=True,
                    think=True,
                    on_token=content_chunks.append,
                )
            finally:
                await ollama.close()
            return result, content_chunks

        result, content_chunks = asyncio.run(run_request())
        payload = json.loads(requests[0].content)

        self.assertTrue(payload["stream"])
        self.assertEqual(content_chunks, ["Die Antwort ", "ist 30."])
        self.assertEqual(result.choices[0].message.content, "Die Antwort ist 30.")
        self.assertEqual(result.choices[0].message.thinking, "Intern weiter")
        self.assertEqual(result.done_reason, "stop")

    def test_chat_tools_and_local_embeddings_use_ollama_api(self):
        requests = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path == "/api/chat":
                return httpx.Response(
                    200,
                    json={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "search_knowledge_base",
                                        "arguments": {"query": "Krankengeld"},
                                    }
                                }
                            ],
                        }
                    },
                )
            if request.url.path == "/api/embed":
                return httpx.Response(
                    200,
                    json={"embeddings": [[0.25, 0.75], [0.5, 0.5]]},
                )
            return httpx.Response(404)

        async def run_requests():
            client = httpx.AsyncClient(
                base_url="http://localhost:11434",
                transport=httpx.MockTransport(handler),
            )
            ollama = LocalOllamaClient(
                "http://localhost:11434",
                "qwen3:4b",
                client=client,
            )
            embedding_client = LocalOllamaClient(
                "http://localhost:11434",
                "nomic-embed-text",
                client=client,
            )
            try:
                chat_result = await ollama.chat_completion(
                    model="qwen3:4b",
                    messages=[{"role": "user", "content": "Krankengeld?"}],
                    tools=[
                        {
                            "type": "function",
                            "function": {"name": "search_knowledge_base"},
                        }
                    ],
                    max_tokens=120,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                embeddings = await embedding_client.feature_extraction(
                    ["Text 1", "Text 2"],
                    normalize=True,
                    truncate=True,
                )
            finally:
                await ollama.close()
                await embedding_client.close()
            return chat_result, embeddings

        chat_result, embeddings = asyncio.run(run_requests())

        self.assertEqual(
            chat_result.choices[0].message.tool_calls[0].function.name,
            "search_knowledge_base",
        )
        self.assertEqual(
            chat_result.choices[0].message.tool_calls[0].function.arguments,
            {"query": "Krankengeld"},
        )
        self.assertEqual(embeddings, [[0.25, 0.75], [0.5, 0.5]])
        self.assertEqual([request.url.path for request in requests], [
            "/api/chat",
            "/api/embed",
        ])
        chat_payload = json.loads(requests[0].content)
        self.assertEqual(chat_payload["model"], "qwen3:4b")
        self.assertFalse(chat_payload["stream"])
        self.assertFalse(chat_payload["think"])
        self.assertEqual(chat_payload["format"], {"type": "json_object"})
        embed_payload = json.loads(requests[1].content)
        self.assertEqual(embed_payload["model"], "nomic-embed-text")

    def test_local_ollama_errors_do_not_fall_back_to_cloud(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "model not found"})

        async def run_request():
            client = httpx.AsyncClient(
                base_url="http://localhost:11434",
                transport=httpx.MockTransport(handler),
            )
            ollama = LocalOllamaClient(
                "http://localhost:11434",
                "missing-model",
                client=client,
            )
            try:
                await ollama.chat_completion(messages=[])
            finally:
                await ollama.close()

        with self.assertRaisesRegex(LocalModelError, "HTTP 404") as error:
            asyncio.run(run_request())
        self.assertIn("model not found", str(error.exception))


if __name__ == "__main__":
    unittest.main()
