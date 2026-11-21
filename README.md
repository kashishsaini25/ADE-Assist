# ADE-Assist: Drug Safety RAG Assistant

A retrieval-augmented question-answering system over FDA drug labels (openFDA). It returns short, **source-cited** answers and refuses when the documents don't contain the answer.

> Educational project. Not medical advice.

**Live demo:** `https://ade-assist-2qpdmzvqlinqrddpf3a4fe.streamlit.app/`

## Problem
Drug labels are long and hard to scan. ADE-Assist lets you ask a natural-language question (for example, "Which antibiotic can cause tendon rupture?") and get a grounded answer with the exact passages it used.

## Architecture
```
openFDA labels -> clean -> chunk (200 words, 40 overlap, drug+section metadata)
      -> embeddings (MiniLM) -> FAISS   \
      -> BM25 index                      > Reciprocal Rank Fusion -> top-k passages
                                         /
question ------------------------------/
      -> Gemini (answer ONLY from passages, cite [n], else refuse) -> answer + sources
```

## Tech stack
Python, openFDA API, sentence-transformers, FAISS, rank_bm25, Gemini API, Streamlit.

## Evaluation
Hand-written question set: **32 questions**, each with a known drug and one or more acceptable label sections.

- **drug hit@k**: at least one of the top-k passages comes from the correct drug.
- **drug+section hit@k**: at least one of the top-k passages comes from the correct drug *and* an acceptable section.

Retrieval comparison (29 drug labels, 931 chunks):

| Retrieval | drug hit@3 | drug+section hit@3 | drug hit@5 | drug+section hit@5 |
|---|---|---|---|---|
| Dense only | 0.938 | 0.875 | 0.938 | **0.906** |
| BM25 only | 0.906 | 0.750 | 0.906 | 0.781 |
| Hybrid (RRF) | 0.906 | 0.844 | 0.938 | 0.875 |

Hybrid retrieval is clearly better than BM25 alone at finding the right section (+9 points at k=5). Dense retrieval is about as good on this set: its 3-point lead over hybrid at k=5 is one question out of 32, so the difference is within noise.

Chunk-size ablation (hybrid, k=5):

| Chunk size (words) | Chunks | drug hit@5 | drug+section hit@5 |
|---|---|---|---|
| 120 | 1426 | 0.844 | 0.781 |
| **200 (default)** | 931 | 0.938 | 0.875 |
| 350 | 537 | 0.938 | 0.875 |

Very small chunks (120 words) lose accuracy. 200 and 350 score the same, so 200 is kept because it gives shorter, more precise cited passages.

Out-of-scope refusal test: **4/4** questions correctly refused (for example, general-knowledge and creative-writing requests).

Error analysis: 4 of 32 questions miss at hybrid k=5. Two are retrieval failures (for example, "calcium channel blocker ... swelling in the ankles" never surfaces amlodipine, because the question names a drug class rather than the drug). One is a drug-interaction question where retrieval returns other drugs' interaction sections. One (metformin / lactic acidosis) retrieves the correct drug but from a section not listed in the expected answer.

## Run locally
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
streamlit run app.py
```
Rebuild the index by running the Colab notebook, which produces `chunks.json` and `embeddings.npy`.

## Limitations
- Covers 29 common drugs, not the whole FDA database.
- Label text can be outdated and varies between manufacturers.
- Evaluation set is small (32 questions) and hand-written, so small differences between methods are not statistically meaningful.
- Generation faithfulness was only spot-checked, not measured automatically.
- Questions that describe a drug class instead of naming a drug can retrieve the wrong drug.
- Depends on the Gemini API, which can be temporarily unavailable; the app retries and falls back across models.

## Future work
Link to ADEGuard (predict adverse-event severity, then explain it from labels), add a reranker, query rewriting for drug-class questions, and automatic faithfulness scoring.
