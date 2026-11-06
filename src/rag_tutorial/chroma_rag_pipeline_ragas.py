"""
1. Einleitung & Zielsetzung
Eine Retrieval-Augmented-Generation-Pipeline (RAG) verbindet ein Sprachmodell (LLM) mit einer
externen Wissensbasis, um fundierte Antworten auf Basis eigener Dokumente zu generieren und
Halluzinationen zu minimieren. Dieses Dokument enthält eine Schritt-für-Schritt-Anleitung sowie das
vollständige Python-Skript, das folgende Komponenten vereint: 
• 
• 
• 
• 
Vektorspeicherung (ChromaDB): Speichert zerlegte Textabschnitte (Chunks) als numerische
Embeddings.
Semantische & Hybride Suche: Kombiniert Vektor-Ähnlichkeitssuche mit lexikalischer Keyword
Suche (BM25) über einen EnsembleRetriever, um sowohl inhaltliche Konzepte als auch exakte
Begriffe (z. B. Fehlercodes) zuverlässig zu finden.
LCEL-RAG-Chain: Verkettet Retrieval, Prompt-Formatierung und Antwortgenerierung in LangChain.
Automatisierte Evaluierung (Ragas): Bewertet Retrieval-Qualität und Antworttreue objektiv anhand
eines Test-Datasets

2. Anleitung zur Einrichtung und Ausführung

Schritt 1: Python-Umgebung & Abhängigkeiten installieren
Stellen Sie sicher, dass Python 3.10 oder neuer installiert ist. Installieren Sie alle benötigten Pakete für
LangChain, ChromaDB, BM25 und Ragas über den folgenden Befehl:
pip install langchain langchain-chroma langchain-community langchain-openai langchain
text-splitters rank_bm25 ragas datasets pandas 

Schritt 2: OpenAI API-Key konfigurieren
Das Skript nutzt text-embedding-3-small für Embeddings sowie gpt-4o-mini für die
Antwortgenerierung und als Judge-LLM in Ragas. Setzen Sie Ihren API-Key als Umgebungsvariable vor
dem Start:
export OPENAI_API_KEY="sk-ihr-api-key"   # Linux / macOS
set OPENAI_API_KEY="sk-ihr-api-key"      # Windows (CMD) 

Schritt 3: Skript speichern und ausführen
1. Speichern Sie den Python-Code aus Abschnitt 4 unter dem Dateinamen 
rag_pipeline_with_ragas.py.
Starten Sie das Skript im Terminal mit python rag_pipeline_with_ragas.py
2. Starten Sie das Skript im Terminal mit python rag_pipeline_with_ragas.py.
3. Beim Ausführen legt ChromaDB automatisch das lokale Verzeichnis ./chroma_db an, verarbeitet die
Beispieldokumente, führt die 4 Testfragen aus und gibt die Ragas-Scores auf der Konsole aus.

Schritt 4: Anpassung an eigene Daten
Suchmodus wechseln: Setzen Sie in build_rag_pipeline(...) den Parameter 
search_type="semantic" für reine Vektorsuche oder search_type="hybrid" für die
kombinierte Suche.
Gewichtung steuern: Über weights=(0.5, 0.5) legen Sie fest, wie stark semantische Suche
(erster Wert) und BM25-Keyword-Suche (zweiter Wert) gewichtet werden.
Eigene Dokumente laden: Ersetzen Sie sample_docs durch LangChain Document Loader (z. B. 
PyPDFLoader) und passen Sie eval_test_cases an Ihre eigenen Fragen und Referenzantworten
an.

3. Übersicht der Ragas-Evaluierungsmetriken
Ragas bewertet jede Metrik auf einer Skala von 0.0 bis 1.0 (je höher der Wert, desto besser die Pipeline
Qualität):
Metrik              Komponente          Beschreibung
Faithfulness        Generator           Prüft, ob alle Aussagen der generierten Antwort durch die abgerufenen Kontexte belegt sind (verhindertHalluzinationen).
Response Relevancy  Generator           Misst, wie direkt, präzise und vollständig die Antwort die gestellte Nutzerfrage beantwortet.
Context Precision   Retriever           Bewertet, ob relevante Dokumente vom Retriever auf den obersten Rängen platziert wurden.
Context Recall      Retriever           Prüft, ob alle Fakten aus der Referenzantwort
(ground_truth) in den abgerufenen Chunks enthalten sind.
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ragas-Imports für die Evaluierung
from pydantic import SecretStr
from ragas import EvaluationDataset, MultiTurnSample, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

load_dotenv()

api_key = SecretStr(os.getenv("OPENROUTER_API_KEY", ""))
if not api_key.get_secret_value():
    raise RuntimeError("OPENROUTER_API_KEY is not set")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-4o-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        model=LLM_MODEL,
        temperature=0,
    )


def make_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        model=EMBEDDING_MODEL,
        check_embedding_ctx_length=False,
    )


def build_rag_pipeline(
    documents: list[Document],
    collection_name: str = "rag_collection",
    persist_directory: str = "./chroma_db",
    search_type: str = "hybrid",
    weights: tuple[float, float] = (0.5, 0.5),
    k: int = 3,
) -> dict[str, Any]:
    """
    Erstellt eine vollständige RAG-Pipeline mit ChromaDB, semantischem und hybridem Retriever
    sowie einer LCEL-RAG-Chain.
    """
    # 1. Text-Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    
    # 2. Embeddings & Vector Store (ChromaDB)
    embeddings = make_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    # 3. Semantischer Retriever (Vektorsuche)
    semantic_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    # 4. Hybrider Retriever (Vektorsuche + BM25 Keyword-Suche)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=list(weights),
    )

    active_retriever = (  hybrid_retriever if search_type == "hybrid" else semantic_retriever)

     # 5. Prompt, LLM und RAG-Chain (LCEL)
    prompt = ChatPromptTemplate.from_template(
        """Beantworte die folgende Frage präzise und ausschließlich auf Basis des 
bereitgestellten Kontexts.
Wenn die Antwort nicht im Kontext enthalten ist, sage ehrlich, dass du es nicht weißt.
Kontext:
{context}
Frage: {question}
Antwort:"""
    )
    llm = make_llm()
    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
        {
            "context": active_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return {
        "vector_store": vector_store,
        "semantic_retriever": semantic_retriever,
        "hybrid_retriever": hybrid_retriever,
        "active_retriever": active_retriever,
        "rag_chain": rag_chain,
    }

def evaluate_rag_pipeline(
    pipeline: dict[str, Any],
    test_cases: list[dict[str, str]],
) -> Any:
    """
    Evaluiert die bestehende RAG-Pipeline (Retriever + Chain) mit Ragas.
    """
    retriever = pipeline["active_retriever"]
    rag_chain = pipeline["rag_chain"]
    samples: list[SingleTurnSample | MultiTurnSample] = []
    # Pipeline für jeden Testfall ausführen und Kontexte sowie Antworten sammeln
    for case in test_cases:
        question = case["question"]
        ground_truth = case["ground_truth"]
        retrieved_docs = retriever.invoke(question)
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        response = rag_chain.invoke(question)
        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=retrieved_contexts,
                response=response,
                reference=ground_truth,
            )
        )
    eval_dataset = EvaluationDataset(samples=samples)

     # Judge-LLM und Embeddings für Ragas wrappen
    evaluator_llm = LangchainLLMWrapper(make_llm())
    evaluator_embeddings = LangchainEmbeddingsWrapper(make_embeddings())
    # Die vier Ragas-Kernmetriken definieren
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]
    # Evaluierung starten
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )
    return results

def run():
    # 5 Beispieldokumente
        sample_docs = [
            Document(
                page_content=(
                    "Eine RAG-Pipeline (Retrieval-Augmented Generation) kombiniert eine "
                    "Vektordatenbank wie ChromaDB mit einem Sprachmodell, um Halluzinationen "
                    "durch fundiertes Kontextwissen zu reduzieren."
                ),
                metadata={"source": "architektur_guide.pdf", "page": 1},
            ),
            Document(
                page_content=(
                    "Fehlercode E-402 tritt auf, wenn der ChromaDB-Server nach 30 Sekunden "
                    "nicht antwortet. Prüfen Sie die Netzwerkverbindung auf Port 8000."
                ),
                metadata={"source": "troubleshooting.pdf", "page": 4},
            ),
            Document(
                page_content=(
                    "Für optimales Text-Splitting empfiehlt sich eine Chunk-Größe von "
                    "500 Zeichen mit einem Overlap von 50 Zeichen, damit Satzgrenzen erhalten bleiben."
                ),
                metadata={"source": "best_practices.pdf", "page": 2},
            ),
            Document(
                page_content=(
                    "Fehlercode E-509 signalisiert eine Rate-Limit-Überschreitung bei der "
                    "Embedding-API. Implementieren Sie Exponential Backoff für automatische Retries."
                ),
                metadata={"source": "troubleshooting.pdf", "page": 7},
            ),
            Document(
                page_content=(
                    "In DSGVO-konformen Produktionsumgebungen sollte ChromaDB lokal betrieben "
                    "und personenbezogene Daten vor dem Embedding anonymisiert werden."
                ),
                metadata={"source": "compliance_richtlinie.pdf", "page": 1},
            ),
        ]
        # Golden Dataset (Testfragen + erwartete Referenzantworten)
        eval_test_cases = [
            {
                "question": "Was bedeutet der Fehlercode E-402 und welchen Port sollte man prüfen?",
                "ground_truth": "Fehlercode E-402 bedeutet, dass der ChromaDB-Server nach 30 Sekunden nicht antwortet. Man sollte Port 8000 prüfen.",
            },
            {
                "question": "Welche Chunk-Größe und welcher Overlap werden für Text-Splitting empfohlen?",
                "ground_truth": "Empfohlen wird eine Chunk-Größe von 500 Zeichen mit einem Overlap von 50 Zeichen.",
            },
            {
                "question": "Wie löst man den Fehlercode E-509?",
                "ground_truth": "Fehlercode E-509 ist eine Rate-Limit-Überschreitung bei der Embedding-API, die durch Exponential Backoff gelöst wird.",
            },
            {
                "question": "Was gilt für DSGVO-konforme Produktionsumgebungen?",
                "ground_truth": "ChromaDB sollte lokal betrieben werden und personenbezogene Daten müssen vor dem Embedding anonymisiert werden.",
            },
        ]
        # Pipeline bauen (hybride Suche)
        pipeline = build_rag_pipeline(
            documents=sample_docs,
            search_type="hybrid",
            k=2,
        )
    
         # Ragas-Evaluierung durchführen
        ragas_result = evaluate_rag_pipeline(pipeline, eval_test_cases)
        print("\n--- Ragas Gesamt-Scores ---")
        print(ragas_result)
        print("\n--- Detailergebnisse pro Frage ---")
        df = ragas_result.to_pandas()
        print(df.drop(columns=["retrieved_contexts"], errors="ignore").to_string())


if __name__ == "__main__":
    run()
    