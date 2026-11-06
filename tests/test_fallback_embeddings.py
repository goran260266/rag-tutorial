"""Testet FallbackEmbeddings und RAGSystem direkt aus dem Notebook 13_ChromaDB_RetrievalQA-chain."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

NOTEBOOK = (
    Path(__file__).parents[1]
    / "src"
    / "rag_tutorial"
    / "13_ChromaDB_RetrievalQA-chain.ipynb"
)


class FakeEmbeddings(Embeddings):
    def __init__(self, dim, fail=False):
        self._inner = DeterministicFakeEmbedding(size=dim)
        self.fail = fail
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("503 Service Unavailable")
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        self.calls += 1
        if self.fail:
            raise RuntimeError("503 Service Unavailable")
        return self._inner.embed_query(text)


def _load_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    imports = next(c for c in code if "import os" in c and "RetrievalQA" in c)
    rag = next(c for c in code if "class FallbackEmbeddings" in c)
    return imports, rag


@pytest.fixture
def ns():
    imports, rag = _load_cells()
    namespace = {"__name__": "notebook"}
    exec(imports, namespace)
    exec(rag, namespace)
    return namespace


@pytest.fixture
def make_rag(ns, tmp_path):
    """Baut ein RAGSystem mit Fake-Modellen (primary 8-dim, backup 4-dim)."""

    def _make(primary_fails=False, backup_fails=False):
        primary = FakeEmbeddings(8, fail=primary_fails)
        backup = FakeEmbeddings(4, fail=backup_fails)
        ns["GoogleGenerativeAIEmbeddings"] = lambda **kwargs: primary
        ns["OpenAIEmbeddings"] = lambda **kwargs: backup
        rag = ns["RAGSystem"](persist_directory=str(tmp_path / "db"))
        return rag, primary, backup

    return _make


def test_uses_primary_when_available(ns):
    primary, backup = FakeEmbeddings(8), FakeEmbeddings(4)
    emb = ns["FallbackEmbeddings"](primary, backup)

    assert len(emb.embed_query("x")) == 8
    assert len(emb.embed_documents(["a", "b"])) == 2
    assert emb.active == "primary"
    assert backup.calls == 0


def test_falls_back_for_query_and_documents(ns):
    primary, backup = FakeEmbeddings(8, fail=True), FakeEmbeddings(4)
    emb = ns["FallbackEmbeddings"](primary, backup)

    assert len(emb.embed_query("x")) == 4
    assert emb.active == "backup"
    assert [len(v) for v in emb.embed_documents(["a", "b"])] == [4, 4]


def test_recovers_when_primary_comes_back(ns):
    primary, backup = FakeEmbeddings(8, fail=True), FakeEmbeddings(4)
    emb = ns["FallbackEmbeddings"](primary, backup)
    emb.embed_query("x")
    assert emb.active == "backup"

    primary.fail = False
    assert len(emb.embed_query("x")) == 8
    assert emb.active == "primary"


def test_raises_when_both_fail(ns):
    emb = ns["FallbackEmbeddings"](
        FakeEmbeddings(8, fail=True), FakeEmbeddings(4, fail=True)
    )
    with pytest.raises(RuntimeError, match="503"):
        emb.embed_query("x")
    with pytest.raises(RuntimeError, match="503"):
        emb.embed_documents(["a"])


def test_rag_system_uses_primary_collection(make_rag):
    rag, primary, backup = make_rag()

    assert rag.primary_embeddings is primary
    assert rag.backup_embeddings is backup
    assert rag.collection_name == "rag_primary"


def test_rag_system_uses_backup_collection_on_primary_failure(make_rag):
    rag, _, _ = make_rag(primary_fails=True)

    assert rag.collection_name == "rag_backup"


def test_rag_system_init_fails_when_both_down(make_rag):
    with pytest.raises(RuntimeError):
        make_rag(primary_fails=True, backup_fails=True)


def test_vector_store_roundtrip_with_backup(make_rag):
    rag, _, _ = make_rag(primary_fails=True)
    chunks = [Document(page_content=f"Chunk {i}") for i in range(3)]

    rag.create_vector_store(chunks)
    rag.load_vector_store()

    assert rag.vector_store._collection.count() == 3
    assert rag.vector_store._collection.name == "rag_backup"
    assert len(rag.vector_store.similarity_search("Chunk 1", k=1)) == 1


def test_models_with_different_dimensions_do_not_clash(make_rag):
    """Index mit primary (8-dim) anlegen, danach fÃ¤llt primary aus: eigene, leere Backup-Collection."""
    rag_primary, _, _ = make_rag()
    rag_primary.create_vector_store([Document(page_content="Chunk A")])

    rag_backup, _, _ = make_rag(primary_fails=True)
    rag_backup.load_vector_store()

    assert rag_backup.vector_store._collection.count() == 0
    rag_backup.create_vector_store([Document(page_content="Chunk B")])
    assert rag_backup.vector_store._collection.count() == 1


class _FakeLLM:
    def __init__(self, name, fail=False, **kwargs):
        self.name, self.fail, self.kwargs = name, fail, kwargs


class _FakeQA:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, inputs):
        if self.llm.fail:
            raise RuntimeError("503 UNAVAILABLE")
        return {"result": f"answer from {self.llm.name}", "source_documents": []}


@pytest.fixture
def rag_with_fake_llms(make_rag, ns):
    def _make(gemini_fails):
        rag, _, _ = make_rag()
        rag.retriever = object()
        ns["ChatGoogleGenerativeAI"] = lambda **kw: _FakeLLM("gemini", fail=gemini_fails, **kw)
        ns["ChatOpenAI"] = lambda **kw: _FakeLLM("backup", **kw)
        ns["RetrievalQA"] = SimpleNamespace(
            from_chain_type=lambda llm, **kw: _FakeQA(llm)
        )
        return rag

    return _make


def test_query_uses_gemini_when_available(rag_with_fake_llms):
    rag = rag_with_fake_llms(gemini_fails=False)

    assert rag.query("q")["answer"] == "answer from gemini"


def test_query_falls_back_to_backup_llm_on_503(rag_with_fake_llms):
    rag = rag_with_fake_llms(gemini_fails=True)

    assert rag.query("q")["answer"] == "answer from backup"


def test_backup_chain_uses_openrouter(rag_with_fake_llms):
    rag = rag_with_fake_llms(gemini_fails=False)

    llm = rag.create_qa_chain(backup=True).llm

    assert llm.name == "backup"
    assert llm.kwargs["base_url"] == "https://openrouter.ai/api/v1"
