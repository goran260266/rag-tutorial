# rag-tutorial

Ein Basis-Projekt für ein Retrieval-Augmented Generation (RAG) System mit LangChain und dem Paketmanager `uv`.

## Projektstruktur

```text
C:\Artificial Intelligence\rag_tutorial\
├── .venv/                      # Virtuelle Python-Umgebung
├── .env                        # Umgebungsvariablen (API-Keys)
├── .gitignore                  # Von Git ignorierte Dateien
├── .python-version             # Python-Version für uv
├── chroma_db/                  # Persistierte ChromaDB-Daten (Projektwurzel)
├── coverage.json               # Coverage-Bericht (temporär)
├── documents/                  # Beispieldokumente (txt, html) für Loader und Splitting
├── src/
│   └── rag_tutorial/
│       ├── __init__.py                     # Hauptmodul des Tutorials
│       ├── 01_document_loader.ipynb        # Dokumente laden
│       ├── 02_text_splitting.ipynb         # Texte in Chunks aufteilen
│       ├── 03_choosing_embedding_models.ipynb  # Embedding-Modelle auswählen
│       ├── 04_similarity_calculations.ipynb    # Ähnlichkeitsberechnungen
│       ├── 07_Quadrant_vectorstore.ipynb   # Vektordatenbank Qdrant
│       ├── 8_Qdrant_retriever-filter.ipynb # Qdrant-Retriever mit Metadatenfilter
│       ├── 09_Qdrant_RAG-chain.ipynb       # RAG-Pipeline mit Qdrant
│       ├── 10_Qdrant_LLM-judge.ipynb       # Evaluierung (Qdrant) mit LLM als Richter
│       ├── 11_DeepEval.ipynb               # Evaluierung mit DeepEval
│       ├── 12_ChromaDB_vectorStore.ipynb   # Vektordatenbank ChromaDB
│       ├── 13_ChromaDB_RetrievalQA-chain.ipynb  # RAG-Pipeline (RetrievalQA) mit ChromaDB
│       ├── 14_ChromaDB_Eval_LLM-Judge.ipynb     # Evaluierung (ChromaDB) mit LLM als Richter
│       ├── 15_Pinacone.ipynb               # Vektordatenbank Pinecone
│       ├── 16_FAISS_vector_database.ipynb  # Vektordatenbank FAISS
│       ├── Workshop_RAG_build_eval.ipynb   # Workshop: RAG aufbauen und evaluieren
│       ├── chroma_rag_pipeline_ragas.py    # RAG-Pipeline mit RAGAS-Evaluierung
│       ├── clean_text.py                   # Textbereinigung
│       ├── rag_utils.py                    # Hilfsfunktionen für RAG
│       ├── vector_embeddings_plots.py      # Visualisierung von Embeddings
│       ├── visualize_chunks.py             # Visualisierung von Chunks
│       ├── Metriken bei RAG-Evaluierung.md # Dokumentation der Evaluierungsmetriken
│       ├── RAGAS_Improving.md              # Hinweise zur Verbesserung der RAGAS-Werte
│       ├── pinecone_bm25_values.json       # BM25-Werte für Pinecone (Hybrid-Suche)
│       ├── py.typed                        # Marker für Typisierung
│       ├── chroma_db/                      # Persistierte ChromaDB-Daten
│       ├── my_rag_db/                      # Persistierte ChromaDB-Daten (RAG)
│       ├── faiss_index/                    # Persistierter FAISS-Index
│       ├── data/                           # Daten für die Notebooks
│       └── img/                            # Abbildungen
├── tests/
│   ├── test_fallback_embeddings.py  # Tests für Fallback-Embeddings
│   └── test_rag.py                  # Tests für die RAG-Funktionen
├── pyproject.toml              # Projektkonfiguration und Abhängigkeiten
├── uv.lock                     # Lock-Datei für exakte Paketversionen
└── README.md                   # Projektdokumentation
```

## Setup-Anleitung

Führen Sie diese Schritte nacheinander im Terminal von Visual Studio Code aus, um das Projekt frisch aufzusetzen und zu konfigurieren:

### 1. Projektverzeichnis bereinigen und initialisieren
Löschen Sie alte Restdateien, um Konflikte zu vermeiden, und initialisieren Sie die Library-Struktur:
```bash
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .venv, src, tests, pyproject.toml, uv.lock, README.md, .python-version
uv init --lib
```

### 2. Testverzeichnis anlegen
Erstellen Sie manuell den Ordner für Unit-Tests und eine erste leere Testdatei:
```bash
New-Item -ItemType Directory -Path "tests"
New-Item -ItemType File -Path "tests\test_rag.py"
```

### 3. Virtuelle Umgebung erstellen und aktivieren
Erzeugen Sie die isolierte Python-Umgebung und aktivieren Sie diese in Ihrer Windows PowerShell:
```bash
uv venv
.venv\Scripts\Activate.ps1
```

### 4. Abhängigkeiten installieren
Synchronisieren Sie das Projekt, um alle in der `pyproject.toml` definierten Pakete herunterzuladen:
```bash
uv sync
```

---

## Funktionstest

Sie können überprüfen, ob das Projekt korrekt importiert werden kann und die virtuelle Umgebung aktiv ist, indem Sie die Standardfunktion `hello()` aus dem Terminal heraus aufrufen:

```bash
uv run python -c "import rag_tutorial; print(rag_tutorial.hello())"
```

**Erwartete Ausgabe:**
```text
Hello from rag-tutorial!
```
### Unit-Tests ausführen
Führen Sie die im `tests/`-Ordner hinterlegten Unit-Tests mit `pytest` aus:
```bash
uv run pytest
```

### Testabdeckung (Coverage) messen
Um zu sehen, wie viel Prozent Ihres Codes durch Tests abgedeckt sind, fügen Sie das Tool hinzu und starten den Report:
```bash
uv add --dev pytest-cov
uv run pytest --cov=rag_tutorial --cov-report=term-missing
```

## Aufräumen & Git-Vorbereitung

Sollten durch das Coverage-Tool oder Ihre IDE temporäre Dateien (wie `coverage.json`) erzeugt werden, können Sie diese löschen und eine `.gitignore`-Datei anlegen, damit temporäre Python-Dateien und Umgebungsvariablen nicht im Projekt stören:

```bash
# Temporäre JSON-Berichte löschen
Remove-Item "coverage.json" -ErrorAction SilentlyContinue

# Automatisch eine .gitignore-Datei für Python & uv erstellen
uv run python -c "
with open('.gitignore', 'w') as f:
    f.write('.venv/\n__pycache__/\n*.pyc\n.coverage\ncoverage.json\nhtmlcov/\n.pytest_cache/\n.env\n')
"
```