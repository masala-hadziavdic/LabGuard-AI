"""LabGuard AI - RAGAS evaluation (RAGAS 0.4.3).

This evaluation environment is intentionally separate from the main app.
Generation and judging both use Mistral's OpenAI-compatible API.
Embeddings for RAGAS use the same local Sentence Transformer as FAISS retrieval.
"""

import csv
import json
import math
import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI, AsyncOpenAI
from langchain_core.embeddings import Embeddings

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
)

ROOT = Path(__file__).resolve().parent
RAG_DIR = ROOT / "rag"
INDEX_PATH = RAG_DIR / "faiss_index.bin"
METADATA_PATH = RAG_DIR / "metadata.pkl"
OUTPUT_DIR = ROOT / "rag-evaluation-results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
GENERATION_MODEL = "mistral-small-latest"
EVALUATOR_MODEL = "mistral-small-latest"
GENERATION_PAUSE_SECONDS = 2.0

# One RAGAS worker avoids bursts of simultaneous requests to the Mistral API.
RUN_CONFIG = RunConfig(
    timeout=180,
    max_workers=1,
    max_retries=4,
    max_wait=60,
    log_tenacity=True,
)

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



class LocalSentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible wrapper around the local retrieval embedding model."""

    def __init__(self, model):
        # IMPORTANT: do not name this attribute `model`.
        # RAGAS 0.4.3 LangchainEmbeddingsWrapper reads `.model` for telemetry
        # and expects a string. SentenceTransformer stores a model object.
        self.encoder = model

    def embed_documents(self, texts):
        vectors = self.encoder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return vectors.tolist()

    def embed_query(self, text):
        vector = self.encoder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return vector.tolist()


def safe_mean(values):
    clean = []
    for value in values:
        try:
            number = float(value)
            if math.isfinite(number):
                clean.append(number)
        except (TypeError, ValueError):
            continue
    return float(np.mean(clean)) if clean else None


def main():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is missing. Add it to GitHub Actions secrets."
        )

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Expected rag/faiss_index.bin and rag/metadata.pkl in the repository."
        )

    print("Loading FAISS index and metadata...")
    index = faiss.read_index(str(INDEX_PATH))
    with METADATA_PATH.open("rb") as file:
        metadata = pickle.load(file)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"FAISS/metadata mismatch: {index.ntotal} vectors vs "
            f"{len(metadata)} metadata records."
        )
    if index.ntotal == 0:
        raise ValueError("The FAISS index is empty.")

    print(f"Loaded {index.ntotal} vectors and {len(metadata)} metadata records.")
    print(f"Loading embedding model: {MODEL_NAME}")
    embedding_model = SentenceTransformer(MODEL_NAME)

    def faiss_search(question, top_k=TOP_K):
        query_vector = embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = index.search(query_vector, min(top_k, index.ntotal))
        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = metadata[int(idx)].copy()
            item["similarity_score"] = float(score)
            retrieved.append(item)
        return retrieved

    # OpenAI-compatible Mistral client for producing LabGuard answers.
    generation_client = OpenAI(
        api_key=api_key,
        base_url=MISTRAL_BASE_URL,
        timeout=120.0,
        max_retries=4,
    )

    # Separate asynchronous client for RAGAS judge calls. This avoids
    # ChatMistralAI's LangChain token-usage aggregation path that failed before.
    evaluator_client = AsyncOpenAI(
        api_key=api_key,
        base_url=MISTRAL_BASE_URL,
        timeout=180.0,
        max_retries=4,
    )
    evaluator_llm = llm_factory(
        EVALUATOR_MODEL,
        provider="openai",
        client=evaluator_client,
        max_tokens=4096,
        temperature=0.0,
    )

    local_embeddings = LangchainEmbeddingsWrapper(
        LocalSentenceTransformerEmbeddings(embedding_model),
        run_config=RUN_CONFIG,
    )

    samples = []
    detailed_rows = []
    print(f"Generating answers for {len(EVALUATION_DATA)} questions...")

    for number, item in enumerate(EVALUATION_DATA, start=1):
        question = item["question"]
        retrieved = faiss_search(question, TOP_K)
        contexts = [
            f"[Source: {r.get('source', 'Unknown')} | "
            f"Chunk: {r.get('chunk_id', 'Unknown')}]\n{r.get('text', '')}"
            for r in retrieved
            if r.get("text")
        ]

        prompt = f"""You are LabGuard AI, a laboratory quality document assistant.
Answer the question using ONLY the information in the retrieved documents.
Rules:
- Do not invent information.
- Do not use external knowledge.
- If the documents do not contain enough information, say that the information is not available.
- Distinguish a suspected cause from a confirmed cause.
- Do not make clinical decisions.
- Answer in the same language as the question.
- Be clear and concise.

QUESTION:
{question}

RETRIEVED DOCUMENTS:
{chr(10).join(contexts)}

ANSWER:
"""

        completion = generation_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": "You are LabGuard AI."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        answer = completion.choices[0].message.content or ""
        if not answer.strip():
            raise RuntimeError(f"Mistral returned an empty answer for question {number}.")

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=[r.get("text", "") for r in retrieved if r.get("text")],
                reference=item["ground_truth"],
            )
        )

        retrieved_sources = [r.get("source", "Unknown") for r in retrieved]
        source_hit = item["expected_source"] in retrieved_sources
        detailed_rows.append({
            "question_number": number,
            "question": question,
            "expected_source": item["expected_source"],
            "retrieved_sources": " | ".join(str(s) for s in retrieved_sources),
            "expected_source_in_top3": source_hit,
            "answer": answer,
        })

        # Save partial results after each answer, so a later failure does not
        # discard the already generated answers.
        with (OUTPUT_DIR / "answers_and_retrieval.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=detailed_rows[0].keys())
            writer.writeheader()
            writer.writerows(detailed_rows)

        print(
            f"[{number}/{len(EVALUATION_DATA)}] Answer generated; "
            f"expected source in top 3: {source_hit}"
        )
        if number < len(EVALUATION_DATA):
            time.sleep(GENERATION_PAUSE_SECONDS)

    evaluation_dataset = EvaluationDataset(samples=samples)
    print("RAGAS dataset ready:", len(samples), "samples")
    print("Running five RAGAS metrics with one worker...")

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=local_embeddings),
        AnswerCorrectness(llm=evaluator_llm, embeddings=local_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        run_config=RUN_CONFIG,
        raise_exceptions=True,
        show_progress=True,
    )

    results_df = result.to_pandas()
    results_df.to_csv(
        OUTPUT_DIR / "ragas_metrics_per_question.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_columns = [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_precision",
        "context_recall",
    ]
    summary = {
        "questions_evaluated": len(samples),
        "top_k": TOP_K,
        "embedding_model": MODEL_NAME,
        "generation_model": GENERATION_MODEL,
        "evaluator_model": EVALUATOR_MODEL,
        "ragas_version": "0.4.3",
        "metrics": {},
        "source_hit_rate_at_3": float(
            np.mean([row["expected_source_in_top3"] for row in detailed_rows])
        ),
    }

    for metric_name in metric_columns:
        if metric_name not in results_df.columns:
            raise KeyError(
                f"Expected RAGAS metric column '{metric_name}' was not returned. "
                f"Available columns: {list(results_df.columns)}"
            )
        summary["metrics"][metric_name] = safe_mean(results_df[metric_name])

    with (OUTPUT_DIR / "ragas_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n=== LABGUARD AI RAGAS SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nEvaluation files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
