"""Central defaults and environment overrides for backend runtime settings."""

from dataclasses import dataclass
import math
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = Path(__file__).resolve().parent
DEFAULT_KNOWLEDGE_DIRECTORY = REPOSITORY_ROOT / "anlagen"
DEFAULT_RAG_INDEX_DIRECTORY = BACKEND_ROOT / ".rag_index"
DEFAULT_SESSION_LOG_DIRECTORY = BACKEND_ROOT / "logs"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120.0
DEFAULT_RAG_TOP_K = 3
DEFAULT_RAG_MINIMUM_SCORE = 0.75
DEFAULT_RAG_EMBEDDING_BATCH_SIZE = 8
DEFAULT_RAG_KEYWORD_MAX_TOKENS = 512
DEFAULT_AGENT_TEMPERATURE = 0.0
DEFAULT_RAG_KEYWORD_TEMPERATURE = 0.1


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
        raise ValueError(f"{name} must be a boolean value.")
    return normalized in {"true", "1", "yes", "on"}


def _directory(name: str, default: str, root: Path) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return (value if value.is_absolute() else root / value).resolve()


def _csv_values(name: str, default: str) -> tuple[str, ...]:
    values = tuple(
        value.strip()
        for value in os.getenv(name, default).split(",")
        if value.strip()
    )
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def _temperature(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or not 0 <= value <= 2:
        raise ValueError(f"{name} must be between 0 and 2.")
    return value


@dataclass(frozen=True)
class BackendConfig:
    repository_root: Path
    knowledge_directory: Path
    rag_index_directory: Path
    session_log_directory: Path
    ollama_host: str
    chat_model: str
    embedding_model: str
    keyword_model: str
    ollama_timeout_seconds: float
    rag_top_k: int
    rag_minimum_score: float
    rag_embedding_batch_size: int
    max_tool_calls_per_turn: int
    max_plan_steps: int
    max_planning_attempts: int
    planning_max_tokens: int
    calculation_max_tokens: int
    final_answer_max_tokens: int
    final_answer_max_chars: int
    agent_temperature: float
    keyword_generation_max_tokens: int
    keyword_generation_temperature: float
    phoenix_enabled: bool
    phoenix_capture_content: bool
    phoenix_collector_endpoint: str
    phoenix_project_name: str
    log_level: str
    cors_origins: tuple[str, ...]
    max_chat_message_chars: int
    max_history_turn_chars: int
    max_history_turns: int

    @classmethod
    def from_env(cls) -> "BackendConfig":
        root = REPOSITORY_ROOT
        timeout = float(
            os.getenv(
                "OLLAMA_TIMEOUT_SECONDS",
                str(DEFAULT_OLLAMA_TIMEOUT_SECONDS),
            )
        )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be positive.")

        rag_minimum_score = float(
            os.getenv("RAG_MINIMUM_SCORE", str(DEFAULT_RAG_MINIMUM_SCORE))
        )
        if not math.isfinite(rag_minimum_score) or not 0 <= rag_minimum_score <= 1:
            raise ValueError("RAG_MINIMUM_SCORE must be between 0 and 1.")

        return cls(
            repository_root=root,
            knowledge_directory=_directory(
                "KNOWLEDGE_DIRECTORY",
                str(DEFAULT_KNOWLEDGE_DIRECTORY),
                root,
            ),
            rag_index_directory=_directory(
                "RAG_INDEX_DIRECTORY",
                str(DEFAULT_RAG_INDEX_DIRECTORY),
                root,
            ),
            session_log_directory=_directory(
                "SESSION_LOG_DIRECTORY",
                str(DEFAULT_SESSION_LOG_DIRECTORY),
                root,
            ),
            ollama_host=os.getenv(
                "OLLAMA_HOST",
                "http://127.0.0.1:11434",
            ).rstrip("/"),
            chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3:4b"),
            embedding_model=os.getenv(
                "OLLAMA_EMBEDDING_MODEL",
                "nomic-embed-text",
            ),
            keyword_model=os.getenv(
                "OLLAMA_KEYWORD_MODEL",
                os.getenv("OLLAMA_CHAT_MODEL", "qwen3:4b"),
            ),
            ollama_timeout_seconds=timeout,
            rag_top_k=_positive_int("RAG_TOP_K", DEFAULT_RAG_TOP_K),
            rag_minimum_score=rag_minimum_score,
            rag_embedding_batch_size=_positive_int(
                "RAG_EMBEDDING_BATCH_SIZE",
                DEFAULT_RAG_EMBEDDING_BATCH_SIZE,
            ),
            max_tool_calls_per_turn=_positive_int(
                "AGENT_MAX_TOOL_CALLS",
                8,
            ),
            max_plan_steps=_positive_int("AGENT_MAX_PLAN_STEPS", 8),
            max_planning_attempts=_positive_int(
                "AGENT_MAX_PLANNING_ATTEMPTS",
                3,
            ),
            planning_max_tokens=_positive_int(
                "AGENT_PLANNING_MAX_TOKENS",
                2048,
            ),
            calculation_max_tokens=_positive_int(
                "AGENT_CALCULATION_MAX_TOKENS",
                128,
            ),
            final_answer_max_tokens=_positive_int(
                "AGENT_FINAL_ANSWER_MAX_TOKENS",
                512,
            ),
            final_answer_max_chars=_positive_int(
                "AGENT_FINAL_ANSWER_MAX_CHARS",
                2500,
            ),
            agent_temperature=_temperature(
                "AGENT_TEMPERATURE",
                DEFAULT_AGENT_TEMPERATURE,
            ),
            keyword_generation_max_tokens=_positive_int(
                "RAG_KEYWORD_MAX_TOKENS",
                DEFAULT_RAG_KEYWORD_MAX_TOKENS,
            ),
            keyword_generation_temperature=_temperature(
                "RAG_KEYWORD_TEMPERATURE",
                DEFAULT_RAG_KEYWORD_TEMPERATURE,
            ),
            phoenix_enabled=_boolean("PHOENIX_ENABLED", True),
            phoenix_capture_content=_boolean(
                "PHOENIX_CAPTURE_CONTENT",
                True,
            ),
            phoenix_collector_endpoint=os.getenv(
                "PHOENIX_COLLECTOR_ENDPOINT",
                "http://localhost:4317",
            ),
            phoenix_project_name=os.getenv(
                "PHOENIX_PROJECT_NAME",
                "sbk-demo",
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            cors_origins=_csv_values(
                "CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ),
            max_chat_message_chars=_positive_int(
                "CHAT_MAX_MESSAGE_CHARS",
                1000,
            ),
            max_history_turn_chars=_positive_int(
                "CHAT_MAX_HISTORY_TURN_CHARS",
                4000,
            ),
            max_history_turns=_positive_int(
                "CHAT_MAX_HISTORY_TURNS",
                20,
            ),
        )
