# Verbesserung der RAG-Metriken: Faithfulness, Relevance, Correctness

Hinweis: Im Notebook `Workshop_RAG_build_eval.ipynb` sind das LLM-Judge-Verdicts (0/1), keine RAGAS-Scores. Die Hebel sind aber dieselben.

## Allgemein (wirkt auf alle drei)

| Hebel | Effekt |
|---|---|
| **Chunking** (`chunk_size`, `chunk_overlap`, Trennzeichen) | Kleinere, thematisch saubere Chunks liefern präziseren Kontext. Zu große Chunks bringen Rauschen, zu kleine verlieren Zusammenhang. |
| **Embedding-Modell** | Ein besseres oder mehrsprachiges Modell hilft bei deutschen Texten (z. B. `text-embedding-3-large`, `bge-m3`, `multilingual-e5`). |
| **Hybrid Search** | Dense und Sparse (BM25) kombinieren (`FastEmbedSparse`, `RetrievalMode.HYBRID`). |
| **Reranking** | Zuerst z. B. 10 Chunks holen, dann per Cross-Encoder (Cohere Rerank, bge-reranker) auf die besten 3–4 reduzieren. Meist besser als ein reines Erhöhen von `k`. |
| **Metadaten-Filter** | Nach Quelle oder Thema filtern, damit irrelevante Dokumente (z. B. `python-*.txt` bei ML-Fragen) nicht im Kontext landen. |
| **Query-Transformation** | Query Rewriting, Multi-Query oder HyDE, wenn die Frage schlecht zur Dokumentsprache passt. |
| **`k` erhöhen** | Mehr Kontext kann die Faithfulness verbessern oder Rauschen hinzufügen. |
| **Anderes Generator-Modell** | Ein Modell, das Anweisungen besser befolgt, verbessert Antwortqualität und Treue zum Kontext. |

## Faithfulness (Antwort stützt sich nur auf den Kontext)

- Strikter Prompt: „Antworte ausschließlich auf Basis des Kontexts. Wenn die Information fehlt, sage 'Ich weiß es nicht'."
- `temperature=0` setzen.
- Weniger, aber relevantere Chunks verwenden. Zu viel Kontext verleitet zu Halluzinationen.
- Zitate oder Quellenangaben in der Antwort verlangen.

## Answer Relevance (Antwort passt zur Frage)

- Prompt soll verlangen, direkt auf die Frage zu antworten, ohne Abschweifung.
- Fragen vor der Suche umformulieren oder präzisieren.
- Antwortlänge und Format im Prompt vorgeben.
- Irrelevante Chunks per Reranking oder Filter entfernen.

## Correctness (Antwort stimmt mit der Referenz überein)

Hängt am stärksten vom Retrieval ab. Zuerst prüfen, ob der richtige Chunk überhaupt abgerufen wird (Context Recall/Precision):

- Chunk fehlt: Problem im Retrieval (Chunking, Embedding, Hybrid, Reranking).
- Chunk vorhanden, Antwort falsch: Problem im Generator (Prompt, stärkeres LLM).

## Die Evaluation selbst verbessern

- **Testset vergrößern**: Bei 10 Fragen sind die Mittelwerte sehr verrauscht. 50 oder mehr Fragen sind besser.
- **Judge-Prompts schärfen** und Few-Shot-Beispiele ergänzen. Ein schwacher Judge (z. B. `gpt-4.1-nano`) verfälscht die Zahlen.
- **Mehrere Läufe** mitteln oder den Judge mit `temperature=0` betreiben.
- **Judge ≠ RAG-Modell** verwenden, um Selbstbevorzugung zu vermeiden.
- **Referenzen aus den Dokumenten** ableiten. Enthält die Referenz Wissen, das nicht in `documents/` steht, kann Correctness nicht steigen.

## Vorgehen

Immer nur eine Variable ändern, neu bewerten und vergleichen (wie `k=2` gegen `k=4`). Sinnvolle Reihenfolge:

1. Chunking
2. Embedding-Modell
3. Hybrid Search und Reranking
4. Prompt
5. Generator-Modell
