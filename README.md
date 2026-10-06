# Demo Chatbot

> **Hinweis:** Die Anwendung und die folgenden Anleitungen wurden nur unter
> Linux getestet.

## Voraussetzungen

- Python 3.10+
- Node.js 20+ und npm
- [Ollama](https://ollama.com/download)
- `uv` für Phoenix (`uvx`)
- Git und Internetzugang für Installation und Modell-Downloads

## Installation

### Ollama-Modelle

Ollama installieren und die benötigten Modelle laden:

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

### Backend

Im Repository-Root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install uv
```

### Frontend

```bash
cd frontend
npm install
```

## Konfiguration

Anpassbare Einstellungen können in `backend/.env` und `backend/config.py` vorgenommen werden.

## Starten

Jeden Dienst in einem eigenen Terminal starten.

**Ollama**

```bash
ollama serve
```

**Phoenix**

```bash
uvx arize-phoenix serve
```

**Backend** – im Repository-Root

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
```

**Frontend**

```bash
cd frontend
npm run dev
```

## Aufrufen

- **Chat:** URL aus der Vite-Ausgabe, üblicherweise `http://localhost:5173`
- **Health-Check:** `http://localhost:8000/api/health`
- **Phoenix:** `http://localhost:6006`

## Tests

Tests liegen in `backend/tests/`. Pytest in der aktiven virtuellen Umgebung
installieren:

```bash
pip install pytest
```

Tests vom Repository-Root ausführen:

```bash
python -m pytest backend/tests -v
```

Die Agent-Integrationstests benötigen ein laufendes Ollama mit den konfigurierten
Modellen und vorhandene Wissensdokumente. Fehlen diese Voraussetzungen, werden
die Integrationstests übersprungen.
