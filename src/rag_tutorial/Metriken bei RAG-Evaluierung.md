# Metriken bei RAG-Evaluierung

Bei der RAG-Evaluierung (Retrieval-Augmented Generation) bewertet man zwei Hauptbereiche:

1. **Retrieval:** Werden die richtigen Dokumente gefunden?
2. **Generation:** Erzeugt das Modell daraus eine korrekte, passende und belegte Antwort?

## Wichtige Metriken

| Metrik | Bewertet | Bedeutung |
|---|---|---|
| **Context Precision** | Retrieval | Sind die gefundenen Dokumente relevant oder enthält der Kontext viel irrelevante Information? |
| **Context Recall** | Retrieval | Enthält der gefundene Kontext alle Informationen, die für die Antwort nötig sind? |
| **Faithfulness / Groundedness** | Antwort | Ist die Antwort vollständig durch den bereitgestellten Kontext belegbar? |
| **Answer Relevancy** | Antwort | Beantwortet die Antwort tatsächlich die Frage? |
| **Answer Correctness** | Antwort | Ist die Antwort inhaltlich richtig und entspricht sie einer Referenzantwort? |
| **Answer Similarity** | Antwort | Wie ähnlich ist die Antwort einer Referenzantwort, häufig semantisch gemessen? |
| **Citation Correctness** | Quellen | Unterstützen die zitierten Quellen die jeweilige Aussage? |
| **Citation Completeness** | Quellen | Sind alle wichtigen Aussagen mit Quellen belegt? |
| **Latency** | System | Wie lange dauert die Anfrage? |
| **Cost** | System | Wie viele Tokens oder API-Kosten entstehen? |

## Klassische Information-Retrieval-Metriken

Diese Metriken bewerten hauptsächlich den Retriever:

- **Precision@k:** Anteil der relevanten Dokumente unter den ersten `k` Treffern.
- **Recall@k:** Anteil aller relevanten Dokumente, die unter den ersten `k` Treffern gefunden wurden.
- **Hit Rate@k:** Wurde mindestens ein relevantes Dokument unter den ersten `k` Treffern gefunden?
- **MRR (Mean Reciprocal Rank):** Wie weit oben steht das erste relevante Dokument?
- **MAP (Mean Average Precision):** Berücksichtigt die Position mehrerer relevanter Dokumente.
- **nDCG:** Bewertet die Rangfolge, auch wenn Dokumente unterschiedlich relevant sind.

## Context Precision

### Beispiel mit schwarzen und weißen Kugeln

- **Weiße Kugeln** = relevante Dokumente
- **Schwarze Kugeln** = irrelevante Dokumente

Der Retriever liefert fünf Dokumente zurück:

    weiß - schwarz - weiß - weiß - schwarz

Davon sind drei von fünf Dokumenten relevant.

Context Precision = Anzahl relevanter Dokumente / Anzahl aller gefundenen Dokumente = 3 / 5 = 60 %

Die Context Precision beträgt also **60 %**.

Eine bessere Rangfolge wäre beispielsweise:

    weiß - weiß - weiß - schwarz - schwarz

Hier stehen die relevanten Dokumente weiter oben. Bei Metriken wie **Context Precision@k** oder **Average Precision** wird diese Rangfolge zusätzlich berücksichtigt.

**Merksatz:**

> Context Precision fragt: „Wie viele der gefundenen Dokumente sind tatsächlich relevant?“

## Context Recall

### Beispiel mit schwarzen und weißen Kugeln

- **Weiße Kugeln** = alle relevanten Dokumente
- **Schwarze Kugeln** = irrelevante Dokumente

In der gesamten Dokumentensammlung befinden sich:

    weiß - weiß - weiß - weiß - weiß - schwarz - schwarz

Es gibt also fünf relevante weiße Kugeln.

Der Retriever findet:

    weiß - weiß - schwarz - schwarz

Er findet zwei der insgesamt fünf relevanten Kugeln.


Context Recall = gefundene relevante Dokumente /  alle relevanten Dokumente = 2 / 5 = 40 %

Der Context Recall beträgt **40 %**.

**Merksatz:**

> Context Recall fragt: „Wie viele der insgesamt wichtigen Dokumente wurden gefunden?“

Im Unterschied dazu fragt Context Precision: „Wie viele der gefundenen Dokumente sind wichtig?“

## Faithfulness / Groundedness

### Beispiel mit Dokumenten

**Dokumente im Kontext:**

> Die Stadtbibliothek ist montags bis freitags von **9:00 bis 18:00 Uhr** geöffnet. Samstags und sonntags bleibt sie geschlossen.

**Frage:**

> Wann ist die Stadtbibliothek geöffnet?

**Antwort A:**

> Die Stadtbibliothek ist montags bis freitags von 9:00 bis 18:00 Uhr geöffnet und am Wochenende geschlossen.

Diese Antwort ist vollständig durch die Dokumente belegt. Die Faithfulness beziehungsweise Groundedness ist hoch.

Faithfulness = 100 %

**Antwort B:**

> Die Stadtbibliothek ist montags bis freitags von 9:00 bis 18:00 Uhr geöffnet. Außerdem ist sie samstags von 10:00 bis 14:00 Uhr geöffnet.

Der erste Satz steht im Dokument. Der zweite Satz wurde jedoch erfunden und widerspricht dem Dokument. Die Antwort ist deshalb nicht vollständig grounded.

**Merksatz:**

> Faithfulness / Groundedness fragt: „Kann jede Aussage der Antwort durch den bereitgestellten Kontext belegt werden?“

Eine Antwort kann sprachlich gut klingen und trotzdem eine schlechte Faithfulness haben, wenn sie Informationen erfindet.

## Answer Relevancy

### Beispiel mit Dokumenten

**Dokument im Kontext:**

> Die Stadtbibliothek ist montags bis freitags von **9:00 bis 18:00 Uhr** geöffnet.

**Frage:**

> Wann ist die Stadtbibliothek geöffnet?

**Antwort A:**

> Die Stadtbibliothek ist montags bis freitags von 9:00 bis 18:00 Uhr geöffnet.

Diese Antwort beantwortet die Frage direkt. Die Answer Relevancy ist hoch.

**Antwort B:**

> Die Stadtbibliothek bietet Bücher, Zeitschriften und Computerarbeitsplätze an. Sie wurde 1985 gegründet.

Diese Informationen können zwar korrekt sein, beantworten aber nicht die Frage nach den Öffnungszeiten. Die Answer Relevancy ist niedrig.

**Merksatz:**

> Answer Relevancy fragt: „Beantwortet die generierte Antwort genau die gestellte Frage?“

Die Metrik bewertet vor allem die Passgenauigkeit der Antwort, nicht primär, ob jede Aussage durch das Dokument belegt ist. Dafür ist Faithfulness / Groundedness zuständig.

## Answer Correctness

### Beispiel mit Dokumenten

**Dokument im Kontext:**

> Die Stadtbibliothek ist montags bis freitags von **9:00 bis 18:00 Uhr** geöffnet.

**Frage:**

> Wann ist die Stadtbibliothek geöffnet?

**Referenzantwort:**

> Die Stadtbibliothek ist montags bis freitags von 9:00 bis 18:00 Uhr geöffnet.

**Antwort A:**

> Die Stadtbibliothek ist montags bis freitags von 9:00 bis 18:00 Uhr geöffnet.

Die Antwort entspricht der Referenzantwort inhaltlich vollständig. Die Answer Correctness ist hoch.

**Antwort B:**

> Die Stadtbibliothek ist montags bis freitags von 10:00 bis 17:00 Uhr geöffnet.

Die Antwort bezieht sich zwar auf die richtige Frage, enthält aber falsche Öffnungszeiten. Die Answer Correctness ist niedrig.

**Merksatz:**

> Answer Correctness fragt: „Ist die Antwort inhaltlich richtig und entspricht sie der erwarteten Referenzantwort?“

Unterschiede:

- **Answer Relevancy:** Beantwortet die Antwort die richtige Frage?
- **Answer Correctness:** Ist die Antwort auch tatsächlich richtig?
- **Faithfulness:** Ist die Antwort durch die bereitgestellten Dokumente belegt?

## Answer Similarity

### Beispiel mit Dokumenten

**Dokument im Kontext:**

> Die Stadtbibliothek ist montags bis freitags von **9:00 bis 18:00 Uhr** geöffnet.

**Frage:**

> Wann ist die Stadtbibliothek geöffnet?

**Referenzantwort:**

> Die Stadtbibliothek ist werktags von 9:00 bis 18:00 Uhr geöffnet.

**Antwort A:**

> Von Montag bis Freitag kann man die Stadtbibliothek zwischen 9 und 18 Uhr besuchen.

Diese Antwort verwendet andere Wörter, hat aber dieselbe Bedeutung. Die Answer Similarity ist hoch.

**Antwort B:**

> Die Bibliothek ist nur vormittags geöffnet.

Diese Antwort hat eine deutlich andere Bedeutung. Die Answer Similarity ist niedrig.

**Merksatz:**

> Answer Similarity fragt: „Wie ähnlich ist die generierte Antwort der Referenzantwort in ihrer Bedeutung?“

Die Metrik misst häufig die semantische Ähnlichkeit mithilfe von Embeddings. Sie bewertet jedoch nicht automatisch, ob die Antwort durch die Dokumente belegt ist. Dafür ist Faithfulness / Groundedness zuständig.

## Citation Correctness

### Beispiel mit Dokumenten

**Dokument 1:**

> Die Stadtbibliothek ist montags bis freitags von **9:00 bis 18:00 Uhr** geöffnet.

**Dokument 2:**

> Samstags und sonntags bleibt die Stadtbibliothek geschlossen.

**Frage:**

> Wann ist die Stadtbibliothek geöffnet?

**Antwort A:**

> Die Stadtbibliothek ist montags bis freitags von 9:00 bis 18:00 Uhr geöffnet. **[Dokument 1]**

Die Quelle belegt die Aussage tatsächlich. Die Citation Correctness ist hoch.

**Antwort B:**

> Die Stadtbibliothek ist montags bis freitags von 9:00 bis 18:00 Uhr geöffnet. **[Dokument 2]**

Die Aussage ist zwar richtig, aber Dokument 2 belegt nur die Schließung am Wochenende. Die zitierte Quelle passt nicht zur Aussage. Die Citation Correctness ist niedrig.

**Merksatz:**

> Citation Correctness fragt: „Unterstützt die zitierte Quelle genau die Aussage, für die sie angegeben wurde?“

Unterschied zu Citation Completeness:

- **Citation Correctness:** Ist die vorhandene Quelle passend?
- **Citation Completeness:** Wurden alle Aussagen zitiert?

## Citation Completeness

### Beispiel mit Dokumenten

**Dokument 1:**

> Die Stadtbibliothek ist montags bis freitags von **9:00 bis 18:00 Uhr** geöffnet.

**Dokument 2:**

> Die Bibliothek bietet kostenloses WLAN an.

**Antwort:**

> Die Stadtbibliothek ist montags bis freitags von 9:00 bis 18:00 Uhr geöffnet. **[Dokument 1]** Außerdem bietet sie kostenloses WLAN an.

Die erste Aussage ist mit einer Quelle belegt. Die zweite Aussage ist zwar durch Dokument 2 belegbar, enthält aber keine Quellenangabe.

Wenn beide Aussagen gleich wichtig sind:


Citation Completeness = belegte wichtige Aussagen / alle wichtigen Aussagen = 1 / 2 = 50 %

Die Citation Completeness beträgt in diesem Beispiel **50 %**.

**Merksatz:**

> Citation Completeness fragt: „Sind alle wichtigen Aussagen der Antwort mit passenden Quellen versehen?“

Unterschied zu Citation Correctness:

- **Completeness:** Wurden alle Aussagen zitiert?
- **Correctness:** Belegt die jeweilige zitierte Quelle die Aussage tatsächlich?

## Bekannte Frameworks

Für RAG-Evaluierung werden häufig verwendet:

- **Ragas:** unter anderem `faithfulness`, `answer_relevancy`, `context_precision` und `context_recall`
- **DeepEval:** unter anderem Faithfulness, Answer Relevancy und Contextual Relevancy
- **TruLens:** unter anderem Groundedness, Context Relevance und Answer Relevance
- **LangSmith:** eigene Datensätze, menschliche Bewertungen und LLM-as-a-Judge

## Zusammenfassung

Eine einzige Metrik reicht nicht aus. Ein gutes RAG-System sollte:

1. relevante Dokumente finden (**Context Precision**),
2. möglichst alle benötigten Informationen finden (**Context Recall**),
3. die Antwort auf den Kontext stützen (**Faithfulness / Groundedness**),
4. die Frage direkt beantworten (**Answer Relevancy**),
5. inhaltlich richtig sein (**Answer Correctness**),
6. der Referenzantwort semantisch ähneln (**Answer Similarity**),
7. passende Quellen angeben (**Citation Correctness**) und
8. alle wichtigen Aussagen mit Quellen versehen (**Citation Completeness**).
