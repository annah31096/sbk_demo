"""Load Markdown knowledge-base documents and their metadata."""

import re
from dataclasses import dataclass
from pathlib import Path


FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)
METADATA_FIELD = re.compile(r"^([a-zA-Z_]+):\s*(.*?)\s*$")


@dataclass(frozen=True)
class MarkdownDocument:
    content: str
    metadata: dict[str, str]


def load_markdown_documents(directory: Path) -> list[MarkdownDocument]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Markdown knowledge directory not found: {directory}")

    documents = []
    for path in sorted(directory.rglob("*.md")):
        raw_content = path.read_text(encoding="utf-8")
        metadata: dict[str, str] = {
            "source": path.relative_to(directory).as_posix(),
            "title": path.stem,
        }
        match = FRONTMATTER.match(raw_content)
        if match:
            for line in match.group(1).splitlines():
                field = METADATA_FIELD.match(line)
                if field:
                    metadata[field.group(1)] = field.group(2).strip("\"'")
            metadata["title"] = metadata.pop("titel", metadata["title"])
            raw_content = raw_content[match.end() :]

        content = raw_content.strip()
        if content:
            documents.append(MarkdownDocument(content=content, metadata=metadata))

    if not documents:
        raise ValueError(f"No non-empty Markdown files found in {directory}.")
    return documents
