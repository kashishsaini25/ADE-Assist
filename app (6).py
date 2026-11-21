import json
import os

import numpy as np
import streamlit as st

from rag_core import HybridRetriever, answer_question

st.set_page_config(page_title="ADE-Assist", page_icon="💊", layout="wide")


@st.cache_resource(show_spinner="Loading drug-label index...")
def load_retriever():
    with open("chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load("embeddings.npy")
    return HybridRetriever(chunks, embeddings=embeddings)


def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


st.title("💊 ADE-Assist: Drug Safety RAG Assistant")
st.caption(
    "Answers come only from FDA drug labels (openFDA) and always cite their source. "
    "Educational use only. This is not medical advice."
)

retriever = load_retriever()
api_key = get_api_key()

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Retrieval mode", ["hybrid", "dense", "bm25"], index=0)
    k = st.slider("Passages to retrieve (k)", 2, 10, 5)
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password")
    st.markdown("**Try:**")
    st.markdown("- Which antibiotic can cause tendon rupture?\n"
                "- Does sertraline carry a suicidality warning?\n"
                "- Can metformin cause lactic acidosis?")

question = st.text_input("Ask a drug-safety question")

if st.button("Ask") and question.strip():
    if not api_key:
        st.error("Please provide a Gemini API key in the sidebar.")
    else:
        with st.spinner("Searching labels and generating answer..."):
            try:
                result = answer_question(question, retriever, k=k, mode=mode, api_key=api_key)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
            else:
                if result["refused"]:
                    st.warning(result["answer"])
                else:
                    st.markdown(result["answer"])
                with st.expander("Sources used"):
                    for i, h in enumerate(result["sources"], 1):
                        st.markdown(f"**[{i}] {h['drug'].title()} | {h['section_name']}** "
                                    f"(score {h['score']})")
                        st.write(h["text"])
