"""ADE-Assist core: data fetch (openFDA), chunking, hybrid retrieval, grounded generation."""
import os
import random
import re
import time

import numpy as np
import requests

OPENFDA_URL = "https://api.fda.gov/drug/label.json"

SECTION_FIELDS = {
    "boxed_warning": "Boxed Warning",
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "warnings": "Warnings",
    "precautions": "Precautions",
    "adverse_reactions": "Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "overdosage": "Overdosage",
    "do_not_use": "Do Not Use",
    "ask_doctor": "Ask a Doctor Before Use",
    "stop_use": "Stop Use and Ask a Doctor",
}

REFUSAL = "I couldn't find this in the provided documents."
MODELS = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-flash-latest", "gemini-3.5-flash-lite"]

# 1. Data: fetch + chunk

def fetch_label(drug, tries=3):
    """Fetch the most complete openFDA drug label for a generic name."""
    params = {"search": f'openfda.generic_name:"{drug}"', "limit": 10}
    for _ in range(tries):
        try:
            r = requests.get(OPENFDA_URL, params=params, timeout=30)
            if r.status_code == 200:
                results = r.json().get("results", [])
                if not results:
                    return None
                return max(
                    results,
                    key=lambda x: sum(len(" ".join(x.get(f, []))) for f in SECTION_FIELDS),
                )
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2)
    return None


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def chunk_words(text, size=200, overlap=40):
    """Split text into overlapping word windows."""
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(labels, size=200, overlap=40, max_words=1500):
    """labels: {drug_name: openFDA label dict}. Returns a list of chunk dicts with metadata."""
    chunks = []
    for drug, label in labels.items():
        set_id = label.get("set_id", "")
        for field, section_name in SECTION_FIELDS.items():
            parts = label.get(field)
            if not parts:
                continue
            text = clean_text(" ".join(parts))
            text = " ".join(text.split()[:max_words])
            for piece in chunk_words(text, size, overlap):
                chunks.append({
                    "id": len(chunks),
                    "drug": drug,
                    "section": field,
                    "section_name": section_name,
                    "set_id": set_id,
                    "text": piece,
                })
    return chunks

# 2. Retrieval: dense (FAISS) + sparse (BM25) + Reciprocal Rank Fusion

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self, chunks, embedder=None, embeddings=None,
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.embedder = embedder or SentenceTransformer(model_name)

        texts = [f"{c['drug']} - {c['section_name']}: {c['text']}" for c in chunks]

        if embeddings is None:
            embeddings = self.embedder.encode(
                texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
            )
        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query, k=5, mode="hybrid", pool=30):
        """mode: 'dense' | 'bm25' | 'hybrid'"""
        if mode not in ("dense", "bm25", "hybrid"):
            raise ValueError("mode must be dense, bm25 or hybrid")
        pool = min(pool, len(self.chunks))
        rankings = []
        if mode in ("dense", "hybrid"):
            q = self.embedder.encode([query], normalize_embeddings=True)
            _, idx = self.index.search(np.asarray(q, dtype="float32"), pool)
            rankings.append([int(i) for i in idx[0] if i >= 0])
        if mode in ("bm25", "hybrid"):
            scores = self.bm25.get_scores(tokenize(query))
            rankings.append([int(i) for i in np.argsort(scores)[::-1][:pool]])

        fused = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank + 1)
        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [dict(self.chunks[i], score=round(s, 4)) for i, s in top]

# 3. Generation: grounded answer with citations + refusal
def build_prompt(question, hits):
    context = "\n\n".join(
        f"[{i + 1}] Drug: {h['drug']} | Section: {h['section_name']}\n{h['text']}"
        for i, h in enumerate(hits)
    )
    return f"""You are ADE-Assist, a drug-safety information assistant.

Rules:
1. Answer ONLY using the CONTEXT below. Do not use outside knowledge.
2. Cite the supporting passages like [1], [2] right after each claim.
3. If the context does not contain the answer, reply exactly: "{REFUSAL}"
4. Be concise (at most 6 sentences). Do not give personal medical advice, personal dosing, or diagnoses.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def generate(prompt, api_key=None, models=None, retries=4):
    """Call Gemini with exponential backoff; skip to the next model on permanent errors."""
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=key)
    errors = {}
    for model in (models or MODELS):
        for attempt in range(retries):
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                return (resp.text or "").strip()
            except Exception as e:
                errors[model] = str(e)[:120]
                msg = str(e)
                # permanent errors: retrying this model is pointless, go to the next one
                if any(w in msg for w in ("404", "NOT_FOUND", "403", "PERMISSION_DENIED",
                                          "INVALID_ARGUMENT", "API key")):
                    break
                # transient (503 UNAVAILABLE, 429, timeouts): back off 2s, 4s, 8s... + jitter
                time.sleep(min(2 ** (attempt + 1), 30) + random.random())
    details = "; ".join(f"{m}: {err}" for m, err in errors.items())
    raise RuntimeError(f"All Gemini models failed. Per-model errors -> {details}")


def answer_question(question, retriever, k=5, mode="hybrid", api_key=None):
    hits = retriever.search(question, k=k, mode=mode)
    text = generate(build_prompt(question, hits), api_key=api_key)
    return {"answer": text, "sources": hits, "refused": REFUSAL.lower() in text.lower()}

# 4. Evaluation helpers

def evaluate_retrieval(retriever, eval_set, k=5, mode="hybrid"):
    """eval_set items: {"q": str, "drug": str, "sections": [section_field, ...]}"""
    drug_hits = section_hits = 0
    for ex in eval_set:
        hits = retriever.search(ex["q"], k=k, mode=mode)
        right_drug = [h for h in hits if h["drug"] == ex["drug"]]
        if right_drug:
            drug_hits += 1
        if any(h["section"] in ex["sections"] for h in right_drug):
            section_hits += 1
    n = len(eval_set)
    return {
        "mode": mode,
        "k": k,
        "drug_hit@k": round(drug_hits / n, 3),
        "drug+section_hit@k": round(section_hits / n, 3),
    }
