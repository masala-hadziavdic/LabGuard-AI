import csv
import json
import os
import pickle
import time
from pathlib import Path
import faiss
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_mistralai import ChatMistralAI
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
RAG_DIR = ROOT / "rag"

INDEX_PATH = RAG_DIR / "faiss_index.bin"
METADATA_PATH = RAG_DIR / "metadata.pkl"

OUTPUT_DIR = ROOT / "rag-evaluation-results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3

# Limitation de la concurrence et gestion des erreurs temporaires
RUN_CONFIG = RunConfig(
    timeout=180,
    max_workers=1,
    max_retries=5,
    max_wait=60,
    log_tenacity=True,)

# Pause entre les appels directs à Mistral
GENERATION_PAUSE = 3

# ============================================================
# DATASET : 15 QUESTIONS D'EVALUATION
# ============================================================

EVALUATION_DATA = [
    {
        "question": "Who is responsible for reviewing and deciding the outcome of an investigation?",
        "expected_source": "Procedure",
        "ground_truth": "The biologist or authorized person reviews the results, evaluates their significance and decides what action should be taken.",
    },
    {
        "question": "What are the responsibilities of laboratory personnel?",
        "expected_source": "Procedure",
        "ground_truth": "Laboratory personnel verify the information related to the reported result.",
    },
    {
        "question": "What are the responsibilities of the quality manager?",
        "expected_source": "Procedure",
        "ground_truth": "The quality manager follows up non-conformities and corrective actions when necessary.",
    },
    {
        "question": "What checks must be performed during the initial verification?",
        "expected_source": "Procedure",
        "ground_truth": "The personnel verify sample identification, test name and unit, recorded value and its consistency with the available data.",
    },
    {
        "question": "What information must be recorded after investigating an anomaly?",
        "expected_source": "Procedure",
        "ground_truth": "The investigation record includes the detection date, test identification, result and unit, detection method, anomaly description, checks performed, suspected or identified cause, decision, actions and closure information.",
    },
    {
        "question": "What information is required to identify an anomaly alert?",
        "expected_source": "Fiche_vierge",
        "ground_truth": "The form includes investigation number, detection date, Sample ID, laboratory, biological test, result, unit, reference interval, dataset outcome and detection method.",
    },
    {
        "question": "Which points must be checked during the initial verification?",
        "expected_source": "Fiche_vierge",
        "ground_truth": "Sample identification, test name and unit, recorded value, reference interval, possible data entry or transcription errors and available complementary information.",
    },
    {
        "question": "Which possible causes of an anomaly should be investigated?",
        "expected_source": "Fiche_vierge",
        "ground_truth": "Possible causes include data entry or transcription errors, sample identification problems, pre-analytical problems, analytical problems, post-analytical problems and possible biological variation.",
    },
    {
        "question": "What possible conclusions can be recorded in the investigation form?",
        "expected_source": "Fiche_vierge",
        "ground_truth": "The form allows recording an unconfirmed anomaly, an unusual value without an identified error, a confirmed data error, an identified non-conformity or the need for further investigation.",
    },
    {
        "question": "What information is recorded in the actions and decision section?",
        "expected_source": "Fiche_vierge",
        "ground_truth": "The form contains the decided action, responsible person, planned date, status and decision of the authorized person.",
    },
    {
        "question": "What glucose result was recorded for Sample ID 23247?",
        "expected_source": "Cas_Glucose",
        "ground_truth": "The recorded glucose result was 10042 mg/dL.",
    },
    {
        "question": "What was the reference interval for glucose in Sample ID 23247?",
        "expected_source": "Cas_Glucose",
        "ground_truth": "The reference interval recorded in the dataset was 70-99 mg/dL.",
    },
    {
        "question": "What was the main suspected cause of the glucose anomaly?",
        "expected_source": "Cas_Glucose",
        "ground_truth": "A data recording or transcription error was suspected, but it was not confirmed.",
    },
    {
        "question": "What verification is necessary to confirm the suspected glucose recording error?",
        "expected_source": "Cas_Glucose",
        "ground_truth": "The original source data must be checked by an authorized person to confirm the suspected recording or transcription error.",
    },
    {
        "question": "Which anomaly detection method identified the glucose result?",
        "expected_source": "Cas_Glucose",
        "ground_truth": "The anomaly was detected using Isolation Forest.",
    },
]

# ============================================================
# LOCAL EMBEDDINGS
# ============================================================

class LocalSentenceTransformerEmbeddings(Embeddings):
    """RAGAS embeddings using the local Sentence Transformer."""

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,)
        return vectors.tolist()

    def embed_query(self, text):
        vector = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,)[0]
        return vector.tolist()

# ============================================================
# CSV SAVING
# ============================================================

def save_detailed_results(rows):
    """Save generated answers and retrieval details."""

    if not rows:
        return

    output_path = OUTPUT_DIR / "answers_and_retrieval.csv"

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",) as f:

        writer = csv.DictWriter(f,
            fieldnames=rows[0].keys(),)

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# SAFE METRIC AVERAGE
# ============================================================

def safe_mean(dataframe, column):
    """Calculate a metric average while ignoring missing values."""

    if column not in dataframe.columns:
        return None

    values = dataframe[column].replace(
        [np.inf, -np.inf],
        np.nan,).dropna()

    if values.empty:
        return None

    return float(values.mean())


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. CHECK API KEY AND FILES
    # --------------------------------------------------------

    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. "
            "Add it as a GitHub Actions secret.")

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Expected rag/faiss_index.bin and rag/metadata.pkl.")

    # --------------------------------------------------------
    # 2. LOAD FAISS INDEX AND METADATA
    # --------------------------------------------------------

    print("\nLoading FAISS index and metadata...")

    index = faiss.read_index(str(INDEX_PATH))

    with METADATA_PATH.open("rb") as f:
        metadata = pickle.load(f)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index/metadata mismatch: "
            f"{index.ntotal} vectors vs "
            f"{len(metadata)} metadata entries.")

    if index.ntotal == 0:
        raise ValueError("The FAISS index is empty.")

    print(f"Loaded {index.ntotal} vectors.")
    print(f"Loaded {len(metadata)} metadata entries.")

    # --------------------------------------------------------
    # 3. LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    print("\nLoading Sentence Transformer...")

    embedding_model = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------------
    # 4. FAISS RETRIEVAL FUNCTION
    # --------------------------------------------------------

    def faiss_search(question, top_k=TOP_K):

        query_embedding = embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,).astype("float32")

        scores, indices = index.search(
            query_embedding,
            min(top_k, index.ntotal),)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            item = metadata[int(idx)].copy()
            item["similarity_score"] = float(score)

            results.append(item)

        return results

    # --------------------------------------------------------
    # 5. INITIALIZE MISTRAL
    # --------------------------------------------------------

    print("\nInitializing Mistral...")

    mistral_llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        max_retries=5,)

    # --------------------------------------------------------
    # 6. GENERATE ANSWERS AND RETRIEVE CONTEXTS
    # --------------------------------------------------------

    retrieval_rows = []
    generation_rows = []
    detailed_rows = []

    print(f"\nRunning retrieval and generation "
        f"for {len(EVALUATION_DATA)} questions...")

    for number, item in enumerate(EVALUATION_DATA, start=1):

        question = item["question"]

        retrieved = faiss_search(question, TOP_K)

        contexts = [(f"[Source: {r.get('source', 'Unknown')} "
                f"| Chunk: {r.get('chunk_id', 'Unknown')}]\n"
                f"{r.get('text', '')}")
            for r in retrieved]

        context_text = "\n\n".join(contexts)

        prompt = f"""You are LabGuard AI, a laboratory quality document assistant.

Answer the question using ONLY the information in the retrieved documents.

Rules:
- Do not invent information.
- Do not use external knowledge.
- If the documents do not contain enough information, clearly say that the information is not available.
- Distinguish a suspected cause from a confirmed cause.
- Do not make clinical decisions.
- Answer in the same language as the question.
- Be clear and concise.

QUESTION:
{question}

RETRIEVED DOCUMENTS:
{context_text}

ANSWER:
"""

        # Generate answer
        answer_response = mistral_llm.invoke(prompt)
        answer = answer_response.content

        if isinstance(answer, list):
            answer = " ".join(
                str(part) for part in answer)

        answer = str(answer)

        # RAGAS retrieval dataset
        retrieval_rows.append({
            "question": question,
            "contexts": [
                r.get("text", "") for r in retrieved],
            "ground_truth": item["ground_truth"],})

        # RAGAS generation dataset
        generation_rows.append({
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": item["ground_truth"],})

        retrieved_sources = [r.get("source") for r in retrieved]

        source_hit = (item["expected_source"] in retrieved_sources)

        detailed_rows.append({
            "question_number": number,
            "question": question,
            "expected_source": item["expected_source"],
            "retrieved_sources": " | ".join(
                str(s) for s in retrieved_sources),
            "expected_source_in_top3": source_hit,
            "answer": answer,})

        # Save intermediate results after every question
        save_detailed_results(detailed_rows)

        print(f"[{number}/{len(EVALUATION_DATA)}] "
            f"Answer generated; "
            f"expected source in top 3: {source_hit}")

        # Pause to reduce API request frequency
        if number < len(EVALUATION_DATA):
            time.sleep(GENERATION_PAUSE)

    # --------------------------------------------------------
    # 7. SAVE DATASETS FOR TRACEABILITY
    # --------------------------------------------------------

    with (OUTPUT_DIR / "retrieval_dataset.json").open(
        "w",
        encoding="utf-8",) as f:
        json.dump(retrieval_rows,
            f,
            indent=2,
            ensure_ascii=False,)

    with (OUTPUT_DIR / "generation_dataset.json").open(
        "w",
        encoding="utf-8",) as f:
        json.dump(generation_rows,
            f,
            indent=2,
            ensure_ascii=False,)

    # --------------------------------------------------------
    # 8. INITIALIZE LOCAL EMBEDDINGS FOR RAGAS
    # --------------------------------------------------------

    local_embeddings = LocalSentenceTransformerEmbeddings(
        embedding_model)

    # --------------------------------------------------------
    # 9. RAGAS RETRIEVAL EVALUATION
    # --------------------------------------------------------

    print("\nRunning RAGAS retrieval metrics...")
    print("Metrics: Context Precision and Context Recall")
    print("Maximum concurrent workers: 1")

    retrieval_dataset = Dataset.from_list(retrieval_rows)

    retrieval_result = evaluate(
        dataset=retrieval_dataset,
        metrics=[context_precision,
            context_recall,],
        llm=mistral_llm,
        is_async=False,
        run_config=RUN_CONFIG,)

    retrieval_df = retrieval_result.to_pandas()

    retrieval_df.to_csv(OUTPUT_DIR / "retrieval_metrics_per_question.csv",
        index=False,)

    print("\nRetrieval evaluation completed.")

    # --------------------------------------------------------
    # 10. RAGAS GENERATION EVALUATION
    # --------------------------------------------------------

    print("\nRunning RAGAS generation metrics...")
    print("Metrics: Faithfulness, Answer Relevancy "
        "and Answer Correctness")
    print("Maximum concurrent workers: 1")

    generation_dataset = Dataset.from_list(generation_rows)

    generation_result = evaluate(
        dataset=generation_dataset,
        metrics=[faithfulness,
            answer_relevancy,
            answer_correctness,],
        llm=mistral_llm,
        embeddings=local_embeddings,
        is_async=False,
        run_config=RUN_CONFIG,)

    generation_df = generation_result.to_pandas()

    generation_df.to_csv(
        OUTPUT_DIR / "generation_metrics_per_question.csv",
        index=False,)

    print("\nGeneration evaluation completed.")

    # --------------------------------------------------------
    # 11. CALCULATE SUMMARY
    # --------------------------------------------------------

    source_hit_rate = float(
        np.mean([row["expected_source_in_top3"]
            for row in detailed_rows]))

    summary = {
        "questions_evaluated": len(EVALUATION_DATA),
        "top_k": TOP_K,
        "embedding_model": MODEL_NAME,
        "mistral_model": "mistral-small-latest",
        "ragas_version": "0.1.9",

        "retrieval_metrics": {
            "context_precision": safe_mean(
                retrieval_df,
                "context_precision",),
            "context_recall": safe_mean(
                retrieval_df,
                "context_recall",),},

        "generation_metrics": {
            "faithfulness": safe_mean(
                generation_df,
                "faithfulness",),
            "answer_relevancy": safe_mean(
                generation_df,
                "answer_relevancy",),
            "answer_correctness": safe_mean(
                generation_df,
                "answer_correctness",),},

        "source_hit_rate_at_3": source_hit_rate,}

    # --------------------------------------------------------
    # 12. SAVE SUMMARY JSON
    # --------------------------------------------------------

    with (OUTPUT_DIR / "ragas_summary.json").open("w",
        encoding="utf-8",) as f:

        json.dump(summary,
            f,
            indent=2,
            ensure_ascii=False,)

    # --------------------------------------------------------
    # 13. DISPLAY FINAL RESULTS
    # --------------------------------------------------------

    print("\n========================================")
    print("       LABGUARD AI RAGAS SUMMARY")
    print("========================================")

    print(json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,))

    print("\nEvaluation files saved in:")
    print(OUTPUT_DIR)

    print("\nRAGAS EVALUATION COMPLETED SUCCESSFULLY.")


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
