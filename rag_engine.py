import os
import pickle
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

# --------------------------------------------------
# 1. Load configuration
# --------------------------------------------------

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found.")


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2")

# --------------------------------------------------
# 3. Load FAISS index
# --------------------------------------------------

@st.cache_resource
def load_faiss_index():

    index = faiss.read_index("rag/faiss_index.bin")

    return index

# --------------------------------------------------
# 4. Load metadata
# --------------------------------------------------

@st.cache_resource
def load_metadata():

    with open("rag/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return metadata

# --------------------------------------------------
# 5. Initialize Mistral
# --------------------------------------------------

@st.cache_resource
def load_mistral():

    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        max_retries=2)


embedding_model = load_embedding_model()
faiss_index = load_faiss_index()
metadata = load_metadata()
mistral_llm = load_mistral()

# --------------------------------------------------
# 6. FAISS retrieval
# --------------------------------------------------

def faiss_search(question, top_k=3):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True).astype("float32")

    k = min(top_k, faiss_index.ntotal)

    scores, indices = faiss_index.search(
        query_embedding,
        k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        item = metadata[idx]

        results.append({
            "score": float(score),
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "text": item["text"]})

    return results

# --------------------------------------------------
# 7. Generate RAG answer
# --------------------------------------------------

def generate_rag_answer(question, contexts):

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are LabGuard AI, a laboratory quality document assistant.

Answer the question using ONLY the information
in the retrieved documents.

Rules:
- Do not invent information.
- Do not use external knowledge.
- If information is unavailable, say so clearly.
- Distinguish suspected causes from confirmed causes.
- Do not make clinical decisions.
- Answer in the same language as the question.
- Be clear and concise.

QUESTION:
{question}

RETRIEVED DOCUMENTS:
{context_text}

ANSWER:
"""

    response = mistral_llm.invoke(prompt)

    return response.content

# --------------------------------------------------
# 8. Complete RAG pipeline
# --------------------------------------------------

def ask_labguard(question, top_k=3):

    retrieved = faiss_search(
        question,
        top_k=top_k)

    if not retrieved:
        return (
            "No relevant document was retrieved.",
            [])

    contexts = [
        f"[Source: {r['source']} | Chunk: {r['chunk_id']}]\n{r['text']}"
        for r in retrieved]

    answer = generate_rag_answer(
        question,
        contexts)

    return answer, retrieved