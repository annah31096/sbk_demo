"""Load prompt text and render its string-template variables."""

from pathlib import Path
from string import Template


PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(name: str, **values: str) -> str:
    path = PROMPTS_DIR / name
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt file is empty: {path}")
    return Template(content).substitute(values)
