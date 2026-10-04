import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.agent import ChatAgent, OUT_OF_SCOPE_REPLY
from backend.markdown_loader import load_markdown_documents
from backend.rag import MarkdownRetriever


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls = []

    async def feature_extraction(self, texts, **kwargs):
        self.calls.append(texts)
        vectors = []
        for text in texts:
            vectors.append(
                [1.0, 0.0]
                if "krankengeld" in text.lower()
                else [0.0, 1.0]
            )
        return np.asarray(vectors, dtype=np.float32)

    async def close(self):
        pass


class FakeKeywordClient:
    def __init__(self) -> None:
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        text = kwargs["messages"][1]["content"].lower()
        if "krankengeld" in text:
            result = {
                "summary": "Informationen zum Anspruch auf Krankengeld.",
                "keywords": ["Krankengeld", "Arbeitsunfähigkeit"],
            }
        else:
            result = {
                "summary": "Informationen zur Familienversicherung.",
                "keywords": ["Familienversicherung", "Angehörige"],
            }
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(result, ensure_ascii=False)
                    )
                )
            ]
        )

    async def close(self):
        pass


class QueuedKeywordClient:
    def __init__(self, responses) -> None:
        self.responses = iter(responses)
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class FakeRetriever:
    def __init__(self) -> None:
        self.queries = []

    async def search(self, query: str):
        self.queries.append(query)
        return [
            {
                "content": "Krankengeld wird nach sechs Wochen gezahlt.",
                "metadata": {
                    "dok_id": "DOK-14",
                    "title": "Krankengeld",
                    "source": "DOK-14_Krankengeld.md",
                    "file_name": "DOK-14_Krankengeld.md",
                },
                "score": 0.9,
                "retrieval_mode": "keywords",
                "summary": "Zusammenfassung",
                "keywords": ["Krankengeld"],
            }
        ]


class EmptyRetriever(FakeRetriever):
    async def search(self, query: str):
        self.queries.append(query)
        return []


class FailingRetriever(FakeRetriever):
    async def search(self, query: str):
        self.queries.append(query)
        raise RuntimeError("Retriever unavailable")


class FakeClient:
    def __init__(
        self,
        thinking: str = "",
        done_reason: str | None = None,
    ) -> None:
        self.requests = []
        self.thinking = thinking
        self.done_reason = done_reason

    async def chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        on_token = kwargs.get("on_token")
        if on_token:
            on_token("Teilantwort ")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Laut [DOK-14: Krankengeld] besteht der Anspruch.",
                        thinking=self.thinking,
                        tool_calls=[],
                    )
                )
            ],
            done_reason=self.done_reason,
        )

    async def close(self):
        pass


class RagTests(unittest.TestCase):
    def test_loader_reads_all_markdown_and_frontmatter(self):
        root = Path(__file__).resolve().parents[2] / "anlagen"
        documents = load_markdown_documents(root)
        self.assertEqual(len(documents), 50)
        krankengeld = next(
            document
            for document in documents
            if document.metadata.get("dok_id") == "DOK-14"
        )
        self.assertEqual(krankengeld.metadata["title"], "Krankengeld")
        self.assertTrue(krankengeld.metadata["source"].endswith(".md"))
        self.assertIn("## Anspruch", krankengeld.content)

    def test_faiss_retriever_returns_relevant_document_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "krankengeld.md").write_text(
                "---\ndok_id: DOK-14\ntitel: Krankengeld\n---\n"
                "# Krankengeld\nAnspruch auf Krankengeld.",
                encoding="utf-8",
            )
            (root / "familie.md").write_text(
                "---\ndok_id: DOK-18\ntitel: Familie\n---\n"
                "# Familie\nFamilienversicherung.",
                encoding="utf-8",
            )
            retriever = MarkdownRetriever(
                root,
                "fake-model",
                keyword_model="fake-model",
                index_directory=root / "index",
                embedding_client=FakeEmbeddingClient(),
                keyword_client=FakeKeywordClient(),
            )
            asyncio.run(retriever.initialize())

            results = asyncio.run(retriever.search("Unbekannte Suchfrage"))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["metadata"]["dok_id"], "DOK-18")
            self.assertEqual(results[0]["retrieval_mode"], "full_text")
            self.assertGreater(results[0]["score"], 0.9)

            cached_embedding_client = FakeEmbeddingClient()
            cached_retriever = MarkdownRetriever(
                root,
                "fake-model",
                keyword_model="fake-model",
                index_directory=root / "index",
                embedding_client=cached_embedding_client,
                keyword_client=FakeKeywordClient(),
            )
            asyncio.run(cached_retriever.initialize())
            self.assertEqual(cached_embedding_client.calls, [])
            cached_results = asyncio.run(
                cached_retriever.search("Unbekannte Suchfrage")
            )
            self.assertEqual(cached_results[0]["metadata"]["dok_id"], "DOK-18")
            self.assertEqual(len(cached_embedding_client.calls), 1)

    def test_keywords_are_persisted_and_searched_before_faiss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "krankengeld.md").write_text(
                "---\ndok_id: DOK-14\ntitel: Krankengeld\n---\n"
                "# Krankengeld\nAnspruch auf Krankengeld.",
                encoding="utf-8",
            )
            (root / "familie.md").write_text(
                "---\ndok_id: DOK-18\ntitel: Familie\n---\n"
                "# Familie\nFamilienversicherung.",
                encoding="utf-8",
            )
            embeddings = FakeEmbeddingClient()
            keyword_model = FakeKeywordClient()
            index_directory = root / "index"
            retriever = MarkdownRetriever(
                root,
                "fake-embedding-model",
                keyword_model="fake-keyword-model",
                index_directory=index_directory,
                embedding_client=embeddings,
                keyword_client=keyword_model,
            )
            asyncio.run(retriever.initialize())

            docstore = json.loads(
                (index_directory / "docstore.json").read_text(encoding="utf-8")
            )
            self.assertEqual(docstore["document_count"], 2)
            self.assertEqual(len(keyword_model.calls), 2)
            entry = next(
                item
                for item in docstore["documents"]
                if item["metadata"]["dok_id"] == "DOK-14"
            )
            self.assertEqual(entry["metadata"]["file_name"], "krankengeld.md")
            self.assertEqual(entry["chunk_id"], "krankengeld.md")
            self.assertIn("Anspruch auf Krankengeld", entry["chunk_text"])
            self.assertEqual(entry["keywords"], ["Krankengeld", "Arbeitsunfähigkeit"])

            embedding_calls_before_keyword_search = len(embeddings.calls)
            keyword_results = asyncio.run(
                retriever.search("Erkläre Krankengeld.")
            )
            self.assertEqual(keyword_results[0]["retrieval_mode"], "keywords")
            self.assertEqual(
                keyword_results[0]["metadata"]["dok_id"],
                "DOK-14",
            )
            self.assertEqual(
                len(embeddings.calls),
                embedding_calls_before_keyword_search,
            )

            fallback_results = asyncio.run(
                retriever.search("Frage ohne passende Schlüsselbegriffe")
            )
            self.assertTrue(
                all(result["retrieval_mode"] == "full_text" for result in fallback_results)
            )
            self.assertEqual(
                len(embeddings.calls),
                embedding_calls_before_keyword_search + 1,
            )

            cached_keywords = FakeKeywordClient()
            cached_retriever = MarkdownRetriever(
                root,
                "fake-embedding-model",
                keyword_model="fake-keyword-model",
                index_directory=index_directory,
                embedding_client=FakeEmbeddingClient(),
                keyword_client=cached_keywords,
            )
            asyncio.run(cached_retriever.initialize())
            self.assertEqual(cached_keywords.calls, [])

    def test_generic_query_words_do_not_trigger_keyword_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "krankengeld.md").write_text(
                "# Krankengeld\nAnspruch auf Krankengeld.",
                encoding="utf-8",
            )
            retriever = MarkdownRetriever(
                root,
                "fake-model",
                embedding_client=FakeEmbeddingClient(),
                keyword_client=FakeKeywordClient(),
            )
            retriever._docstore = [
                {
                    "keywords": ["Leistung", "Krankengeld"],
                    "metadata": {"title": "Krankengeld"},
                    "summary": "Zusammenfassung",
                    "chunk_text": "Anspruch auf Krankengeld.",
                }
            ]

            results = retriever._search_keywords(
                "Welche Leistung gibt es bei einer Raumfahrtversicherung?"
            )

            self.assertEqual(results, [])

    def test_indexing_logs_keyword_progress_and_reuses_completed_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, title in (
                ("one.md", "Krankengeld"),
                ("two.md", "Familienversicherung"),
            ):
                (root / filename).write_text(
                    f"---\ntitel: {title}\n---\n# {title}\nDokumentinhalt.",
                    encoding="utf-8",
                )
            keyword_client = FakeKeywordClient()
            retriever = MarkdownRetriever(
                root,
                "fake-embedding-model",
                keyword_model="fake-keyword-model",
                index_directory=root / "index",
                embedding_client=FakeEmbeddingClient(),
                keyword_client=keyword_client,
            )
            with self.assertLogs("backend.rag", level="INFO") as logs:
                asyncio.run(retriever.initialize())
            messages = "\n".join(logs.output)
            self.assertIn("RAG indexing started: 2 Markdown files", messages)
            self.assertIn("Docstore 1/2: generating local summary and keywords", messages)
            self.assertIn("Docstore 2/2: saved", messages)
            self.assertIn("RAG indexing complete.", messages)
            self.assertEqual(len(keyword_client.calls), 2)

    def test_keyword_generation_retries_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "soziotherapie.md"
            source.write_text(
                "---\ntitel: Soziotherapie\n---\n"
                "# Soziotherapie\nUnterstützung bei schweren Erkrankungen.",
                encoding="utf-8",
            )
            invalid_response = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"summary": "Unvollständige Antwort"',
                            thinking="",
                        )
                    )
                ]
            )
            valid_json = {
                "summary": "Soziotherapie unterstützt Versicherte bei schweren "
                "Erkrankungen.",
                "keywords": ["Soziotherapie", "psychische Erkrankung"],
            }
            valid_response = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(valid_json, ensure_ascii=False),
                            thinking="",
                        )
                    )
                ]
            )
            client = QueuedKeywordClient([invalid_response, valid_response])
            retriever = MarkdownRetriever(
                root,
                "fake-embedding-model",
                keyword_model="fake-keyword-model",
                embedding_client=FakeEmbeddingClient(),
                keyword_client=client,
            )
            document = load_markdown_documents(root)[0]

            summary, keywords = asyncio.run(retriever._summarize_document(document))

            self.assertIn("Soziotherapie unterstützt", summary)
            self.assertEqual(keywords, ["Soziotherapie", "psychische Erkrankung"])
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(client.calls[0]["response_format"]["type"], "object")
            self.assertEqual(client.calls[0]["max_tokens"], 512)

    def test_keyword_generation_reads_json_from_thinking_if_content_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "soziotherapie.md").write_text(
                "# Soziotherapie\nUnterstützung bei schweren Erkrankungen.",
                encoding="utf-8",
            )
            response = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="",
                            thinking=(
                                '{"summary":"Unterstützung bei Erkrankungen.",'
                                '"keywords":["Soziotherapie"]}'
                            ),
                        )
                    )
                ]
            )
            client = QueuedKeywordClient([response])
            retriever = MarkdownRetriever(
                root,
                "fake-embedding-model",
                keyword_model="fake-keyword-model",
                embedding_client=FakeEmbeddingClient(),
                keyword_client=client,
            )

            summary, keywords = asyncio.run(
                retriever._summarize_document(load_markdown_documents(root)[0])
            )

            self.assertEqual(summary, "Unterstützung bei Erkrankungen.")
            self.assertEqual(keywords, ["Soziotherapie"])
            self.assertEqual(len(client.calls), 1)

    def test_krankenkassen_question_uses_rag_and_injects_source_context(self):
        retriever = FakeRetriever()
        fake_client = FakeClient(thinking="Interne Überlegung")
        agent = ChatAgent(
            model="test-model",
            retriever=retriever,
            chat_client=fake_client,
        )

        async def run_agent():
            return [
                event
                async for event in agent.stream(
                    "Wie funktioniert Krankengeld bei der Krankenkasse?"
                )
            ]

        events = asyncio.run(run_agent())
        self.assertEqual(len(retriever.queries), 1)
        self.assertEqual(
            retriever.queries[0],
            "Wie funktioniert Krankengeld bei der Krankenkasse?",
        )
        self.assertEqual(events[0]["type"], "tool_start")
        self.assertEqual(events[0]["name"], "search_knowledge_base")
        self.assertEqual(events[1]["type"], "tool_result")
        self.assertIn("[DOK-14: Krankengeld]", events[1]["output"])
        system_message = fake_client.requests[0]["messages"][0]["content"]
        self.assertIn("nicht vertrauenswürdige Nutzdaten", system_message)
        self.assertIn("fiktive Musterkasse", system_message)
        self.assertTrue(fake_client.requests[0]["think"])
        response_event = next(
            event for event in events if event["type"] == "llm_response"
        )
        self.assertEqual(response_event["thinking"], "Interne Überlegung")
        self.assertEqual(events[-1]["type"], "final")

    def test_follow_up_rag_query_and_model_use_full_chat_context(self):
        retriever = FakeRetriever()
        fake_client = FakeClient()
        agent = ChatAgent(
            model="test-model",
            retriever=retriever,
            chat_client=fake_client,
        )
        history = [
            {"role": "user", "content": "Ich frage zur Zahnreinigung."},
            {
                "role": "assistant",
                "content": "Geht es dir um Kosten oder Voraussetzungen?",
            },
            {"role": "user", "content": "Wie hoch ist der Zuschuss?"},
            {
                "role": "assistant",
                "content": "Welche konkrete Frage hast du dazu?",
            },
        ]

        async def run_agent():
            return [
                event
                async for event in agent.stream("Und bei Kindern?", history=history)
            ]

        events = asyncio.run(run_agent())

        self.assertEqual(
            retriever.queries,
            [
                "Ich frage zur Zahnreinigung. Wie hoch ist der Zuschuss? "
                "Und bei Kindern?"
            ],
        )
        model_messages = fake_client.requests[0]["messages"]
        self.assertIn(
            {"role": "user", "content": "Ich frage zur Zahnreinigung."},
            model_messages,
        )
        self.assertIn(
            {"role": "user", "content": "Wie hoch ist der Zuschuss?"},
            model_messages,
        )
        self.assertIn(
            {"role": "user", "content": "Und bei Kindern?"},
            model_messages,
        )
        self.assertIn(
            "eindeutig beantwortbar ist",
            model_messages[0]["content"],
        )
        self.assertEqual(events[-1]["type"], "final")

    def test_short_follow_up_inherits_gkv_scope_from_user_history(self):
        agent = ChatAgent(
            model="test-model",
            retriever=FakeRetriever(),
            chat_client=FakeClient(),
        )
        state = {
            "messages": [
                {"role": "user", "content": "Frage zu Krankengeld"},
                {"role": "assistant", "content": "Was möchtest du wissen?"},
                {"role": "user", "content": "Voraussetzungen"},
            ]
        }

        update = agent._route_request(state)

        self.assertEqual(
            update["messages"][0]["tool_calls"][0]["function"]["name"],
            "search_knowledge_base",
        )
        self.assertEqual(
            json.loads(
                update["messages"][0]["tool_calls"][0]["function"]["arguments"]
            )["query"],
            "Frage zu Krankengeld Voraussetzungen",
        )

    def test_calculation_about_gkv_topic_uses_calculator_with_context(self):
        agent = ChatAgent(
            model="test-model",
            retriever=FakeRetriever(),
            chat_client=FakeClient(),
        )
        state = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Wie hoch ist bei meinem Einkommen meine jährliche "
                        "Belastungsgrenze?"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Die Grenze beträgt im Regelfall 2 % des jährlichen "
                        "Bruttoeinkommens. [DOK-04: Belastungsgrenze]"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "berechne meine beim einkommen von 1234 euro monatlich, "
                        "ich bin regelfall"
                    ),
                },
            ]
        }

        update = agent._route_request(state)

        self.assertEqual(
            update["messages"][0]["tool_calls"][0]["function"]["name"],
            "calculate",
        )
        expression = json.loads(
            update["messages"][0]["tool_calls"][0]["function"]["arguments"]
        )["expression"]
        self.assertEqual(expression, "1234 * 12 * 2 %")
        from backend.tools import calculate

        self.assertEqual(calculate(expression), "296,16")

        async def run_agent():
            return [
                event
                async for event in agent.stream(
                    state["messages"][-1]["content"],
                    history=state["messages"][:-1],
                )
            ]

        events = asyncio.run(run_agent())
        self.assertEqual(events[0]["type"], "tool_start")
        self.assertEqual(events[0]["name"], "calculate")
        self.assertEqual(events[1]["type"], "tool_result")
        self.assertIn("296,16", events[1]["output"])

    def test_contextual_percentage_calculation_uses_calculator(self):
        agent = ChatAgent(
            model="test-model",
            retriever=FakeRetriever(),
            chat_client=FakeClient(),
        )
        messages = [
            {
                "role": "user",
                "content": "Wie hoch ist meine jährliche Belastungsgrenze?",
            },
            {
                "role": "assistant",
                "content": "Sie beträgt im Regelfall 2 % pro Jahr.",
            },
            {
                "role": "user",
                "content": "Berechne sie für mein Einkommen von 1234 Euro monatlich.",
            },
        ]

        update = agent._route_request({"messages": messages})
        call = update["messages"][0]["tool_calls"][0]

        self.assertEqual(call["function"]["name"], "calculate")
        expression = json.loads(call["function"]["arguments"])["expression"]
        self.assertEqual(expression, "1234 * 12 * 2 %")
        from backend.tools import calculate

        self.assertEqual(calculate(expression), "296,16")

    def test_percentage_calculation_in_current_message_uses_calculator(self):
        agent = ChatAgent(
            model="test-model",
            retriever=FakeRetriever(),
            chat_client=FakeClient(),
        )

        update = agent._route_request(
            {
                "messages": [
                    {"role": "user", "content": "Berechne 2 % von 1234 Euro."}
                ]
            }
        )
        call = update["messages"][0]["tool_calls"][0]

        self.assertEqual(call["function"]["name"], "calculate")
        self.assertEqual(
            json.loads(call["function"]["arguments"])["expression"],
            "1234 * 2 %",
        )

    def test_unrelated_question_after_gkv_history_keeps_original_scope_refusal(self):
        agent = ChatAgent(
            model="test-model",
            retriever=FakeRetriever(),
            chat_client=FakeClient(),
        )
        state = {
            "messages": [
                {"role": "user", "content": "Frage zur Krankengeld-Regelung"},
                {"role": "assistant", "content": "Welche Frage hast du dazu?"},
                {"role": "user", "content": "Erkläre die Quantenmechanik ausführlich."},
            ]
        }

        update = agent._route_request(state)

        self.assertEqual(update["control_action"], "out_of_scope")
        self.assertEqual(update["final_reply"], OUT_OF_SCOPE_REPLY)
        self.assertNotIn("messages", update)

    def test_percentage_question_still_uses_calculator_not_rag(self):
        retriever = FakeRetriever()
        fake_client = FakeClient()
        agent = ChatAgent(
            model="test-model",
            retriever=retriever,
            chat_client=fake_client,
        )

        async def run_agent():
            return [
                event
                async for event in agent.stream("Was sind 15 % von 200?")
            ]

        events = asyncio.run(run_agent())
        self.assertEqual(events[0]["type"], "tool_start")
        self.assertEqual(events[0]["name"], "calculate")
        self.assertIn("30", events[1]["output"])
        self.assertEqual(retriever.queries, [])
        self.assertEqual(events[-1]["type"], "final")

    def test_model_length_limit_is_not_returned_as_a_final_answer(self):
        fake_client = FakeClient(
            thinking="Reasoning stopped mid-sentence",
            done_reason="length",
        )
        agent = ChatAgent(
            model="test-model",
            retriever=FakeRetriever(),
            chat_client=fake_client,
        )
        state = {
            "messages": [{"role": "user", "content": "Erkläre Krankengeld."}],
            "tool_call_count": 0,
            "on_answer_token": lambda token: None,
        }

        async def call_model():
            return await agent._call_model(state)

        update = asyncio.run(call_model())
        response_state = {
            **state,
            "messages": [*state["messages"], *update["messages"]],
            "model_done_reason": update["model_done_reason"],
        }

        self.assertEqual(fake_client.requests[0]["max_tokens"], 4096)
        self.assertTrue(fake_client.requests[0]["think"])
        self.assertEqual(
            update["trace_events"][0]["thinking"],
            "Reasoning stopped mid-sentence",
        )
        with self.assertRaisesRegex(RuntimeError, "nicht vollständig erzeugt"):
            agent._record_model_response(response_state)

    def test_vague_gkv_topic_searches_rag_and_asks_for_clarification(self):
        retriever = FakeRetriever()
        fake_client = FakeClient()
        agent = ChatAgent(
            model="test-model",
            retriever=retriever,
            chat_client=fake_client,
        )

        async def run_agent():
            return [event async for event in agent.stream("Krankengeld")]

        events = asyncio.run(run_agent())

        self.assertEqual(retriever.queries, ["Krankengeld"])
        self.assertEqual(events[0]["type"], "tool_start")
        self.assertEqual(events[1]["type"], "tool_result")
        self.assertIn("Welche konkrete Frage", events[-1]["reply"])
        self.assertIn("Krankengeld", events[-1]["reply"])
        self.assertEqual(fake_client.requests, [])

    def test_question_without_sources_is_not_sent_to_the_language_model(self):
        retriever = EmptyRetriever()
        fake_client = FakeClient()
        agent = ChatAgent(
            model="test-model",
            retriever=retriever,
            chat_client=fake_client,
        )

        async def run_agent():
            return [
                event
                async for event in agent.stream(
                    "Welche Leistung gibt es bei einer Raumfahrtversicherung?"
                )
            ]

        events = asyncio.run(run_agent())

        self.assertEqual(len(retriever.queries), 1)
        self.assertIn("keine passende Quelle", events[-1]["reply"])
        self.assertIn("nichts erfinde", events[-1]["reply"])
        self.assertEqual(fake_client.requests, [])

    def test_out_of_scope_question_is_refused_without_calling_model_or_rag(self):
        retriever = FakeRetriever()
        fake_client = FakeClient()
        agent = ChatAgent(
            model="test-model",
            retriever=retriever,
            chat_client=fake_client,
        )

        async def run_agent():
            return [event async for event in agent.stream("Erkläre Quantenphysik.")]

        events = asyncio.run(run_agent())

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "final")
        self.assertIn("nur Fragen", events[0]["reply"])
        self.assertEqual(retriever.queries, [])
        self.assertEqual(fake_client.requests, [])

    def test_rag_tool_failure_returns_guarded_reply_without_model_generation(self):
        retriever = FailingRetriever()
        fake_client = FakeClient()
        agent = ChatAgent(
            model="test-model",
            retriever=retriever,
            chat_client=fake_client,
        )

        async def run_agent():
            return [
                event
                async for event in agent.stream(
                    "Welche Voraussetzungen gelten für Krankengeld?"
                )
            ]

        events = asyncio.run(run_agent())

        self.assertEqual(len(retriever.queries), 1)
        self.assertEqual(fake_client.requests, [])
        self.assertIn("keine verlässliche Antwort", events[-1]["reply"])

    def test_final_validation_requires_successful_tool_evidence(self):
        agent = ChatAgent(
            model="test-model",
            retriever=FakeRetriever(),
            chat_client=FakeClient(),
        )
        state = {
            "tool_evidence": [],
            "final_reply": "Unbelegte Antwort.",
        }

        result = agent._validate_final(state)

        self.assertEqual(result["control_action"], "missing_tool")
        self.assertIn("keine Sachantwort", result["final_reply"])

    def test_final_validation_rejects_rag_results_without_source_metadata(self):
        agent = ChatAgent(
            model="test-model",
            retriever=FakeRetriever(),
            chat_client=FakeClient(),
        )
        state = {
            "tool_evidence": [
                {
                    "name": "search_knowledge_base",
                    "succeeded": False,
                    "source_count": 0,
                }
            ],
            "final_reply": "Unbelegte Antwort.",
        }

        result = agent._validate_final(state)

        self.assertEqual(result["control_action"], "no_sources")
        self.assertIn("keine passende Quelle", result["final_reply"])

    def test_clarification_answer_reuses_the_topic_from_conversation(self):
        retriever = FakeRetriever()
        fake_client = FakeClient()
        agent = ChatAgent(
            model="test-model",
            retriever=retriever,
            chat_client=fake_client,
        )
        history = [
            {"role": "user", "content": "Zahnreinigung"},
            {
                "role": "assistant",
                "content": (
                    "Geht es dir um Voraussetzungen oder Kosten? "
                    "Welche konkrete Frage hast du dazu?"
                ),
            },
        ]

        async def run_agent():
            return [
                event
                async for event in agent.stream("Zuschuss", history=history)
            ]

        events = asyncio.run(run_agent())

        self.assertEqual(retriever.queries, ["Zahnreinigung Zuschuss"])
        self.assertEqual(len(fake_client.requests), 1)
        self.assertEqual(events[-1]["type"], "final")


if __name__ == "__main__":
    unittest.main()
