from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_tutorial import hello


def test_chroma_from_documents_persists(tmp_path):
    """Der Notebook-Code legt eine persistente Chroma-DB an (ohne persist())."""
    splits = [Document(page_content=f"Chunk {i}") for i in range(5)]
    embeddings = DeterministicFakeEmbedding(size=8)
    persist_dir = str(tmp_path / "chroma_db")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="my_collection",
    )
    assert not hasattr(vectorstore, "persist")

    reloaded = Chroma(
        collection_name="my_collection",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    assert reloaded._collection.count() == len(splits)


def test_hello_returns_correct_string():
    """Überprüft, ob die hello-Funktion den erwarteten Begrüßungstext zurückgibt."""
    # Act (Ausführung)
    result = hello()
    
    # Assert (Überprüfung)
    assert result == "Hello from rag-tutorial!"
