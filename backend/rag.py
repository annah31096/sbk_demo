"""Local Ollama embeddings, keyword docstore, and FAISS retriever."""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend.local_llm import LocalOllamaClient
from backend.markdown_loader import MarkdownDocument, load_markdown_documents

PROMPTS_DIR = Path(__file__).with_name("prompts")
KEYWORD_PROMPT = (PROMPTS_DIR / "build_keywords.txt").read_text(
    encoding="utf-8"
).strip()
KEYWORD_SCHEMA_VERSION = 4
KEYWORD_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["summary", "keywords"],
}
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
GENERIC_SEARCH_TOKENS = {
    "aber",
    "als",
    "am",
    "an",
    "auch",
    "auf",
    "bei",
    "bekomme",
    "das",
    "dass",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eines",
    "es",
    "erstattung",
    "erstattet",
    "für",
    "gibt",
    "haben",
    "hat",
    "ich",
    "im",
    "in",
    "ist",
    "kann",
    "kosten",
    "kostet",
    "krankenkasse",
    "kasse",
    "leistungen",
    "leistung",
    "mit",
    "muss",
    "nach",
    "oder",
    "sind",
    "zahlt",
    "zahlen",
    "und",
    "versicherung",
    "versicherte",
    "vom",
    "von",
    "war",
    "was",
    "welche",
    "welcher",
    "welches",
    "wer",
    "wie",
    "wird",
    "zu",
    "über",
}
logger = logging.getLogger(__name__)


class MarkdownRetriever:
    def __init__(
        self,
        documents_directory: Path,
        embedding_model: str,
        *,
        ollama_host: str = "http://127.0.0.1:11434",
        keyword_model: str | None = None,
        top_k: int = 4,
        minimum_score: float = 0.7,
        index_directory: Path | None = None,
        embedding_client: Any | None = None,
        keyword_client: Any | None = None,
    ) -> None:
        self._embedding_model = embedding_model
        self._keyword_model = keyword_model or embedding_model
        self._ollama_host = ollama_host
        self._documents: list[MarkdownDocument] = load_markdown_documents(
            documents_directory
        )
        self._embedding_client = embedding_client or LocalOllamaClient(
            ollama_host,
            embedding_model,
        )
        self._keyword_client = keyword_client or LocalOllamaClient(
            ollama_host,
            self._keyword_model,
        )
        self._index: faiss.Index | None = None
        self._top_k = top_k
        self._minimum_score = minimum_score
        self._index_directory = index_directory or Path(__file__).with_name(
            ".rag_index"
        )
        self._index_path = self._index_directory / "documents.faiss"
        self._index_metadata_path = self._index_directory / "index_meta.json"
        self._legacy_metadata_path = self._index_directory / "documents.json"
        self._docstore_path = self._index_directory / "docstore.json"
        self._index_fingerprint = self._create_index_fingerprint()
        self._docstore_fingerprint = self._create_docstore_fingerprint()
        self._docstore: list[dict[str, Any]] = []
        self._load_cached_index()

    async def initialize(self) -> None:
        logger.info(
            "RAG indexing started: %d Markdown files, embedding model '%s', "
            "keyword model '%s'.",
            len(self._documents),
            self._embedding_model,
            self._keyword_model,
        )
        await self._load_or_build_docstore()
        if self._index is not None:
            logger.info("Loaded persisted FAISS index from %s", self._index_path)
            logger.info("RAG indexing complete; all persisted files are ready.")
            return

        logger.info(
            "Building FAISS index for %d documents with '%s'.",
            len(self._documents),
            self._embedding_model,
        )
        document_texts = [
            self._embedding_text(document) for document in self._documents
        ]
        document_vectors = []
        for start in range(0, len(document_texts), 8):
            batch_end = min(start + 8, len(document_texts))
            logger.info(
                "Embedding RAG documents %d-%d of %d.",
                start + 1,
                batch_end,
                len(document_texts),
            )
            document_vectors.extend(
                await self._embed(document_texts[start : start + 8])
            )
        vectors = np.asarray(document_vectors, dtype=np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._persist_index(index, vectors.shape[1])
        self._index = index
        logger.info("Built and persisted FAISS index at %s", self._index_path)
        logger.info("RAG indexing complete.")

    @staticmethod
    def _embedding_text(document: MarkdownDocument) -> str:
        return f"{document.metadata['title']}\n{document.content}"

    async def search(self, query: str) -> list[dict[str, Any]]:
        if not self._docstore:
            raise RuntimeError("The Markdown retriever has not been initialized.")
        keyword_results = self._search_keywords(query)
        if keyword_results:
            return keyword_results
        if self._index is None:
            raise RuntimeError("The Markdown retriever has not been initialized.")

        query_vector = np.asarray(
            await self._embed([query]),
            dtype=np.float32,
        )
        scores, indexes = self._index.search(query_vector, self._top_k)
        results = []
        for score, index in zip(scores[0], indexes[0]):
            if index < 0 or float(score) < self._minimum_score:
                continue
            results.append(
                self._search_result(
                    self._docstore[int(index)],
                    score=float(score),
                    retrieval_mode="full_text",
                )
            )
        return results

    async def close(self) -> None:
        await self._embedding_client.close()
        if self._keyword_client is not self._embedding_client:
            await self._keyword_client.close()

    def _create_index_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self._embedding_model.encode("utf-8"))
        digest.update(b"\0ollama-normalized-mean-pooling-v1\0")
        for document in self._documents:
            digest.update(self._embedding_text(document).encode("utf-8"))
            digest.update(b"\0")
            digest.update(
                json.dumps(
                    document.metadata,
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            )
            digest.update(b"\0")
        return digest.hexdigest()

    def _create_docstore_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self._keyword_model.encode("utf-8"))
        digest.update(
            f"\0ollama-keyword-schema-{KEYWORD_SCHEMA_VERSION}\0".encode()
        )
        for document in self._documents:
            digest.update(self._document_hash(document).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _document_hash(document: MarkdownDocument) -> str:
        serialized = json.dumps(
            {"content": document.content, "metadata": document.metadata},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    async def _load_or_build_docstore(self) -> None:
        cached: dict[str, dict[str, Any]] = {}
        if self._docstore_path.is_file():
            try:
                payload = json.loads(self._docstore_path.read_text(encoding="utf-8"))
                if (
                    payload.get("schema_version") in {1, KEYWORD_SCHEMA_VERSION}
                    and payload.get("keyword_model") == self._keyword_model
                ):
                    for entry in payload.get("documents", []):
                        if isinstance(entry, dict) and isinstance(
                            entry.get("source"), str
                        ):
                            migrated_entry = dict(entry)
                            if (
                                "chunk_text" not in migrated_entry
                                and isinstance(migrated_entry.get("content"), str)
                            ):
                                migrated_entry["chunk_text"] = migrated_entry.pop(
                                    "content"
                                )
                            cached[entry["source"]] = migrated_entry
            except (OSError, ValueError, TypeError):
                logger.warning("Persisted docstore is unreadable; rebuilding metadata.")

        logger.info(
            "Docstore: %d cached entries found; checking %d source documents.",
            len(cached),
            len(self._documents),
        )
        entries: list[dict[str, Any]] = []
        for index, document in enumerate(self._documents, start=1):
            source = document.metadata["source"]
            content_hash = self._document_hash(document)
            existing = cached.get(source)
            if (
                existing
                and existing.get("content_hash") == content_hash
                and isinstance(existing.get("summary"), str)
                and isinstance(existing.get("keywords"), list)
                and existing["keywords"]
            ):
                entries.append(existing)
                logger.info(
                    "Docstore %d/%d: reused keywords for %s.",
                    index,
                    len(self._documents),
                    source,
                )
                continue

            logger.info(
                "Docstore %d/%d: generating local summary and keywords for %s.",
                index,
                len(self._documents),
                source,
            )
            try:
                summary, keywords = await self._summarize_document(document)
            except Exception:
                logger.exception(
                    "Docstore keyword generation failed for %s.",
                    source,
                )
                raise
            entries.append(
                self._create_docstore_entry(
                    document,
                    content_hash,
                    summary,
                    keywords,
                )
            )
            self._write_docstore(entries)
            logger.info(
                "Docstore %d/%d: saved %d keywords for %s.",
                index,
                len(self._documents),
                len(keywords),
                source,
            )

        self._docstore = entries
        if len(self._docstore) != len(self._documents):
            raise RuntimeError("The RAG docstore could not be built completely.")
        self._write_docstore(self._docstore)
        logger.info(
            "Docstore ready: %d documents saved to %s.",
            len(self._docstore),
            self._docstore_path,
        )

    async def _summarize_document(
        self,
        document: MarkdownDocument,
    ) -> tuple[str, list[str]]:
        source = document.metadata["source"]
        document_prompt = (
            f"Datei: {source}\n"
            f"Titel: {document.metadata['title']}\n\n"
            f"{document.content}"
        )
        last_error: ValueError | None = None
        for attempt in range(1, 4):
            messages = [
                {"role": "system", "content": KEYWORD_PROMPT},
                {"role": "user", "content": document_prompt},
            ]
            if attempt > 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Deine vorige Antwort war kein gültiges JSON mit "
                            "einem nicht-leeren summary-Text und einer nicht-leeren "
                            "keywords-Liste. Antworte jetzt ausschließlich mit "
                            "einem gültigen JSON-Objekt gemäß dem vorgegebenen Schema."
                        ),
                    }
                )
            logger.info(
                "Keyword generation for %s: attempt %d/3.",
                source,
                attempt,
            )
            response = await self._keyword_client.chat_completion(
                model=self._keyword_model,
                messages=messages,
                max_tokens=512,
                temperature=0.1,
                response_format=KEYWORD_RESPONSE_SCHEMA,
            )
            if not response.choices:
                last_error = ValueError("Das Modell lieferte keine Antwort.")
            else:
                response_message = response.choices[0].message
                content = getattr(response_message, "content", "")
                thinking = getattr(response_message, "thinking", "")
                candidates = [
                    candidate
                    for candidate in (content, thinking)
                    if isinstance(candidate, str) and candidate.strip()
                ]
                last_error = None
                for candidate in candidates:
                    try:
                        return self._parse_keyword_output(candidate)
                    except ValueError as error:
                        last_error = error

            logger.warning(
                "Keyword generation for %s failed on attempt %d/3: %s.",
                source,
                attempt,
                last_error or "Antwort ohne JSON-Inhalt",
            )

        raise RuntimeError(
            f"Das lokale Keyword-Modell lieferte nach 3 Versuchen kein gültiges "
            f"JSON für {source}: {last_error or 'leere Antwort'}."
        ) from last_error

    @staticmethod
    def _parse_keyword_output(content: str) -> tuple[str, list[str]]:
        json_start = content.find("{")
        if json_start < 0:
            raise ValueError("JSON-Objekt fehlt.")
        try:
            parsed, _ = json.JSONDecoder().raw_decode(content[json_start:])
        except json.JSONDecodeError as error:
            raise ValueError("Antwort enthält ungültiges JSON.") from error
        if not isinstance(parsed, dict):
            raise ValueError("JSON-Antwort muss ein Objekt sein.")

        summary = parsed.get("summary")
        keywords = parsed.get("keywords")
        if (
            not isinstance(summary, str)
            or not summary.strip()
            or not isinstance(keywords, list)
            or not keywords
            or not all(
                isinstance(keyword, str) and keyword.strip()
                for keyword in keywords
            )
        ):
            raise ValueError("Zusammenfassung oder Keywords fehlen.")
        return (
            summary.strip(),
            list(dict.fromkeys(keyword.strip() for keyword in keywords)),
        )

    @staticmethod
    def _create_docstore_entry(
        document: MarkdownDocument,
        content_hash: str,
        summary: str,
        keywords: list[str],
    ) -> dict[str, Any]:
        metadata = dict(document.metadata)
        metadata["file_name"] = Path(document.metadata["source"]).name
        return {
            "chunk_id": document.metadata["source"],
            "content_hash": content_hash,
            "source": document.metadata["source"],
            "summary": summary,
            "keywords": keywords,
            "chunk_text": document.content,
            "metadata": metadata,
        }

    @staticmethod
    def _search_result(
        entry: dict[str, Any],
        *,
        score: float,
        retrieval_mode: str,
    ) -> dict[str, Any]:
        return {
            **entry,
            "content": entry["chunk_text"],
            "score": score,
            "retrieval_mode": retrieval_mode,
        }

    def _search_keywords(self, query: str) -> list[dict[str, Any]]:
        query_tokens = {
            token
            for token in TOKEN_PATTERN.findall(query.casefold())
            if token not in GENERIC_SEARCH_TOKENS and len(token) > 2
        }
        ranked = []
        for entry in self._docstore:
            keywords = entry.get("keywords", [])
            exact_matches = sum(
                1
                for keyword in keywords
                if set(TOKEN_PATTERN.findall(keyword.casefold()))
                and set(TOKEN_PATTERN.findall(keyword.casefold()))
                <= query_tokens
            )
            keyword_tokens = {
                token
                for keyword in keywords
                for token in TOKEN_PATTERN.findall(keyword.casefold())
                if len(token) > 2 and token not in GENERIC_SEARCH_TOKENS
            }
            matched_tokens = query_tokens & keyword_tokens
            if not exact_matches and not matched_tokens:
                continue
            score = exact_matches + len(matched_tokens) / max(
                len(keyword_tokens), 1
            )
            ranked.append((score, entry))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            self._search_result(
                entry,
                score=score,
                retrieval_mode="keywords",
            )
            for score, entry in ranked[: self._top_k]
        ]

    def _load_cached_index(self) -> None:
        metadata_path = (
            self._index_metadata_path
            if self._index_metadata_path.is_file()
            else self._legacy_metadata_path
        )
        if not self._index_path.is_file() or not metadata_path.is_file():
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") != self._index_fingerprint:
                logger.info(
                    "Markdown sources or embedding model changed; rebuilding index."
                )
                return
            index = faiss.read_index(str(self._index_path))
            if (
                index.ntotal != len(self._documents)
                or index.d != metadata.get("dimension")
                or metadata.get("document_count") != len(self._documents)
            ):
                logger.warning(
                    "Persisted FAISS index has invalid dimensions; rebuilding."
                )
                return
            self._index = index
            if metadata_path == self._legacy_metadata_path:
                self._persist_index_metadata(metadata)
                self._legacy_metadata_path.unlink(missing_ok=True)
        except (OSError, ValueError, RuntimeError, TypeError, json.JSONDecodeError):
            logger.warning("Persisted FAISS index is unreadable; rebuilding.")

    def _persist_index(self, index: faiss.Index, dimension: int) -> None:
        self._index_directory.mkdir(parents=True, exist_ok=True)
        temporary_index = self._index_path.with_suffix(".faiss.tmp")
        try:
            faiss.write_index(index, str(temporary_index))
            os.replace(temporary_index, self._index_path)
            self._persist_index_metadata(
                {
                    "fingerprint": self._index_fingerprint,
                    "embedding_model": self._embedding_model,
                    "document_count": len(self._documents),
                    "dimension": dimension,
                }
            )
        finally:
            temporary_index.unlink(missing_ok=True)

    def _persist_index_metadata(self, metadata: dict[str, Any]) -> None:
        self._index_directory.mkdir(parents=True, exist_ok=True)
        temporary_metadata = self._index_metadata_path.with_suffix(".json.tmp")
        try:
            temporary_metadata.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary_metadata, self._index_metadata_path)
        finally:
            temporary_metadata.unlink(missing_ok=True)

    def _write_docstore(self, entries: list[dict[str, Any]]) -> None:
        self._index_directory.mkdir(parents=True, exist_ok=True)
        temporary_docstore = self._docstore_path.with_suffix(".json.tmp")
        try:
            temporary_docstore.write_text(
                json.dumps(
                    {
                        "schema_version": KEYWORD_SCHEMA_VERSION,
                        "keyword_model": self._keyword_model,
                        "fingerprint": self._docstore_fingerprint,
                        "document_count": len(self._documents),
                        "documents": entries,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(temporary_docstore, self._docstore_path)
        finally:
            temporary_docstore.unlink(missing_ok=True)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        raw_vectors = await self._embedding_client.feature_extraction(
            texts,
            normalize=True,
            truncate=True,
        )
        vectors = np.asarray(raw_vectors, dtype=np.float32)
        if vectors.ndim == 1 and len(texts) == 1:
            vectors = vectors.reshape(1, -1)
        elif vectors.ndim == 3 and vectors.shape[0] == len(texts):
            vectors = vectors.mean(axis=1)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise RuntimeError("Das lokale Embedding-Modell lieferte ungültige Vektoren.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or np.any(norms == 0):
            raise RuntimeError("Das lokale Embedding-Modell lieferte ungültige Vektoren.")
        vectors = vectors / norms
        return vectors.tolist()
