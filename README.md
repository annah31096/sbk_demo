# Demo Chatbot

A small chatbot demo with a Vite frontend and a FastAPI backend.

## Requirements

- Node.js 20 or newer and npm
- Python 3.10 or newer

## Lokale Modelle einrichten

Installiere [Ollama](https://ollama.com/download), starte den Ollama-Dienst und
lade die Chat-, Embedding- und Keyword-Modelle einmal lokal herunter:

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

Qwen 3 wird für die JSON-Ausgabe der Keyword-Analyse ohne Thinking-Modus
aufgerufen, damit die Antwort im JSON-Inhalt statt nur im Denkfeld landet.
Ungültige Keyword-Antworten werden bis zu zweimal erneut angefordert; JSON im
Thinking-Feld wird ebenfalls erkannt.

Die Standardkonfiguration steht in `backend/.env.example`. Bei Bedarf kannst
du diese Einstellungen in `backend/.env` übernehmen und anpassen:

```dotenv
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_CHAT_MODEL=qwen3:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_KEYWORD_MODEL=qwen3:4b
```

Chat, Embeddings und Keyword-Erstellung laufen lokal über Ollama; ein
Hugging-Face-Token wird nicht benötigt. `OLLAMA_KEYWORD_MODEL` muss ein lokales
Chat-Modell sein, das JSON-Zusammenfassungen ausgeben kann.

## Backend starten

Führe die Befehle im Repository-Root aus:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Die API läuft unter `http://localhost:8000`; die interaktive Dokumentation ist
unter `http://localhost:8000/docs` verfügbar. Beim Backend-Start wird der lokale
FAISS-Index geladen oder erstellt und der Docstore um fehlende Keywords
ergänzt. Dafür muss Ollama laufen und die konfigurierten Modelle müssen
heruntergeladen sein. Der Indexierungsfortschritt für Dokumente, Keywords und
Embeddings wird im Backend-Log ausgegeben.

Der Chatbot ist ein LangGraph-Agent. Prozentfragen rufen das Tool `calculate`
auf. Im Frontend wird nur die finale Agentenantwort angezeigt; Thinking,
Modellantworten und Tool-Aufrufe werden nicht an die Oberfläche gesendet. Bei
laufenden Anfragen zeigt das Frontend einen Spinner; die finale Antwort wird
tokenweise gestreamt. Finale Antworten mit Markdown-Formatierung (Überschriften,
Fett-/Kursivtext, Listen, Zitate und Code) werden im Chat formatiert dargestellt;
HTML aus Modellantworten wird nicht ausgeführt. Prompts liegen in
`backend/prompts/`.

Fragen zur gesetzlichen Krankenversicherung werden über den RAG-Retriever
beantwortet. Beim Start werden Markdown-Dateien unter `anlagen/` samt
Frontmatter-Metadaten geladen, mit dem lokalen Ollama-Modell
`nomic-embed-text` eingebettet und in FAISS indiziert. Die vier relevantesten
Dokumente werden dem Agenten bereitgestellt, der ihre Dokument-IDs und Titel
als Quellen nennen soll. Mit `OLLAMA_EMBEDDING_MODEL` in `backend/.env` lässt
sich ein anderes lokales Ollama-Embedding-Modell einstellen. Die Dokumente
werden nicht weiter aufgeteilt (ein Chunk pro Datei). Beim Erstellen des
Docstores wird jede Markdown-Datei einmal an das lokale Modell
`OLLAMA_KEYWORD_MODEL` gesendet, um eine kurze Zusammenfassung und Suchbegriffe
zu erstellen. Beides wird zusammen mit dem vollständigen Inhalt, Dateinamen und
Frontmatter im Feld `chunk_text` in `backend/.rag_index/docstore.json`
gespeichert. Die Suche prüft zuerst die Keywords und nutzt nur bei keinem
Treffer die semantische FAISS-Suche über den vollständigen Inhalt. Der
FAISS-Index und Docstore werden bei späteren Starts wiederverwendet; neu
generiert werden nur Keywords für neue oder geänderte Dokumente. Lokale
Indexdateien sind von Git ausgeschlossen. Die Dateien beschreiben eine
fiktive Musterkasse und sind keine verbindliche Rechts- oder
Versicherungsberatung.

Bei unklaren GKV-Themen sucht der Agent zunächst passende Anlagen und bittet
dann im Chat um eine konkrete Frage; die gefundenen Dokumenttitel helfen dabei,
mögliche Aspekte einzugrenzen. Die Unterhaltung wird für Rückfragen im Browser
begrenzt mitgeführt. Findet die Suche keine passende Quelle, gibt der Agent
keine generierte Sachantwort aus, sondern weist auf die fehlende Grundlage hin
und bittet um eine Eingrenzung.
Der Agent beantwortet nur Fragen zu den freigegebenen GKV-Themen und explizite
Rechenaufgaben. Andere Sachfragen werden regelbasiert abgewiesen, ohne das LLM
aufzurufen. Vor einer finalen Antwort wird geprüft, dass ein Werkzeug erfolgreich
gelaufen ist und eine GKV-Suche tatsächlich mindestens eine Quelle mit
Dokumentmetadaten geliefert hat. Bei Tool-Fehlern oder fehlenden Quellen wird
stattdessen eine feste, sichere Rückmeldung ausgegeben.
Rückfragen werden anhand des bisherigen Gesprächsverlaufs eingeordnet; das RAG
verwendet die vorherigen Nutzernachrichten zusammen mit der aktuellen Eingabe.
Wenn wesentliche Angaben für eine belegte Antwort fehlen, soll der Agent gezielt
nachfragen, statt eine Annahme zu treffen.

Agentenereignisse werden lokal als JSONL in `backend/logs/` protokolliert,
einschließlich Nutzereingaben, Thinking, Tool-Aufrufen/-Ergebnissen und finalen
Antworten. Pro Browser-Session entsteht eine Datei
`session-<UTC-Zeitstempel>-<session-id>.jsonl`, damit Logs nach Zeitstempel
sortiert werden können; das Verzeichnis ist von Git ausgeschlossen.
Eine Session beginnt bei jedem Neuladen der Chatseite oder über die Schaltfläche
„Sitzung löschen“ neu. Dabei werden Verlauf und Nachrichten im Frontend
zurückgesetzt und eine neue Logdatei-ID verwendet; vorhandene Session-Logs
bleiben erhalten.

## Frontend starten

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Öffne die von Vite angezeigte lokale URL (normalerweise
`http://localhost:5173`). Der Chat verwendet die API unter
`http://localhost:8000`.

## Präsentation

Die interaktive HTML-Präsentation liegt unter `presentation/index.html`.
Öffne die Datei direkt im Browser; navigiere mit den Pfeiltasten oder den
Steuerelementen unten. Die erste Folie zeigt den Agenten-Graphen.

If Node reports `SyntaxError: Unexpected reserved word` on an `await import(...)`
line, the frontend is being started with an outdated Node.js runtime. Check
`node --version` in the same terminal used to start Vite, upgrade to Node.js 20
or newer, and restart the frontend. If you start it from an IDE, check that the
IDE's Node.js interpreter also points to the updated installation.
