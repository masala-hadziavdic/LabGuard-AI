import os
import pickle
from pathlib import Path
from functools import lru_cache
from io import BytesIO
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_mistralai import ChatMistralAI
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_PATH = Path(__file__).resolve().parent

RAG_PATH = BASE_PATH / "rag"

FAISS_PATH = RAG_PATH / "faiss_index.bin"

METADATA_PATH = RAG_PATH / "metadata.pkl"

load_dotenv(BASE_PATH / ".env")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


app = FastAPI(title="LabGuard AI API",
    description=(
        "API for laboratory anomaly documentation "
        "and RAG document assistance."),
    version="1.0.0")


# ============================================================
# 2. LOAD RAG COMPONENTS
# ============================================================

@lru_cache(maxsize=1)
def load_embedding_model():

    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2")


@lru_cache(maxsize=1)
def load_faiss_index():

    if not FAISS_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_PATH}")

    return faiss.read_index(str(FAISS_PATH))


@lru_cache(maxsize=1)
def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}")

    with open(METADATA_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_mistral():

    if not MISTRAL_API_KEY:
        raise ValueError(
            "MISTRAL_API_KEY is missing in .env")

    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        max_retries=2)


# ============================================================
# 3. RAG RETRIEVAL
# ============================================================

def faiss_search(question: str, top_k: int = 3):

    embedding_model = load_embedding_model()
    index = load_faiss_index()
    metadata = load_metadata()

    if index.ntotal == 0:
        return []

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True).astype("float32")

    k = min(top_k, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        result = metadata[idx].copy()
        result["similarity_score"] = float(score)

        results.append(result)

    return results


# ============================================================
# 4. RAG GENERATION
# ============================================================

def generate_rag_answer(question: str, retrieved_docs: list):

    mistral_llm = load_mistral()

    contexts = [(f"[Source: {doc['source']} | "
            f"Chunk: {doc['chunk_id']}]\n"
            f"{doc['text']}")
        for doc in retrieved_docs]

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are LabGuard AI, a laboratory quality document assistant.

Answer using ONLY the retrieved documents.

Rules:
- Do not invent information.
- Do not use external knowledge.
- If information is missing, say so clearly.
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


# ============================================================
# 5. REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):

    question: str = Field(
        min_length=3,
        max_length=2000)

    top_k: int = Field(
        default=3,
        ge=1,
        le=10)


# ============================================================
# 6. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health_check():

    return {"status": "healthy",
        "application": "LabGuard AI API",
        "version": "1.0.0"}


# ============================================================
# 7. RAG ASK ENDPOINT
# ============================================================

@app.post("/ask")
def ask_labguard_api(request: AskRequest):

    try:

        retrieved = faiss_search(
            request.question,
            top_k=request.top_k)

        if not retrieved:

            return {"question": request.question,
                "answer": "No relevant document was retrieved.",
                "sources": []}

        answer = generate_rag_answer(request.question,retrieved)

        sources = [{"source": doc.get("source"),
                "chunk_id": doc.get("chunk_id"),
                "similarity_score": doc.get(
                    "similarity_score"),
                "text": doc.get("text")}
            for doc in retrieved]

        return {"question": request.question,
            "answer": answer,
            "sources": sources}

    except FileNotFoundError as e:

        raise HTTPException(status_code=500,detail=str(e))

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"LabGuard AI error: {type(e).__name__}: {e}")


# ============================================================
# 8. GENERATE GLUCOSE INVESTIGATION WORD DOCUMENT
# ============================================================

def generate_glucose_investigation_form():

    doc = Document()

    # Page layout
    section = doc.sections[0]

    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    # General font
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(10)

    # --------------------------------------------------------
    # FORMATTING FUNCTIONS
    # --------------------------------------------------------

    def add_title(text):

        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

        run = p.add_run(text)
        run.bold = True
        run.font.name = "Arial"
        run.font.size = Pt(16)
        run.font.color.rgb = RGBColor(31, 78, 121)

    def add_section(title):

        p = doc.add_heading(title, level=1)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)

    def add_table(headers, rows):

        table = doc.add_table(rows=1,cols=len(headers))

        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        # Header
        for i, header in enumerate(headers):

            cell = table.rows[0].cells[i]
            cell.text = header

            shading = cell._tc.get_or_add_tcPr()

            shading.append(parse_xml(r'<w:shd {} w:fill="1F4E79"/>'.format(nsdecls("w"))))

            for paragraph in cell.paragraphs:

                for run in paragraph.runs:

                    run.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
                    run.font.size = Pt(9)

        # Data rows
        for row_data in rows:

            cells = table.add_row().cells

            for i, value in enumerate(row_data):

                cells[i].text = str(value)

                cells[i].vertical_alignment = (WD_CELL_VERTICAL_ALIGNMENT.CENTER)

                for paragraph in cells[i].paragraphs:

                    for run in paragraph.runs:

                        run.font.name = "Arial"
                        run.font.size = Pt(9)

        doc.add_paragraph()

        return table

    def add_text(text):

        doc.add_paragraph(text)

    def add_bold_text(text):

        p = doc.add_paragraph()
        p.add_run(text).bold = True

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    add_title("CAS D'ÉTUDE GLUCOSE")

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    run = subtitle.add_run("CAS D'ÉTUDE – DONNÉES DU DATASET")

    run.bold = True
    run.font.size = Pt(11)

    doc.add_paragraph()

    # --------------------------------------------------------
    # 1. IDENTIFICATION DE L'ALERTE
    # --------------------------------------------------------

    add_section("1. Identification de l'alerte")

    add_table(
        ["Information", "Donnée"],
        [
            ("Numéro de l'investigation", "LAB-ANOM-0001"),
            ("Date de détection", "2026-11-11"),
            ("Sample ID", "23247"),
            ("Laboratoire", "Biolab"),
            ("Examen biologique", "Glucose"),
            ("Résultat", "10 042 mg/dL"),
            ("Unité", "mg/dL"),
            ("Intervalle de référence", "70–99 mg/dL"),
            ("Outcome dans le dataset", "Abnormal"),
            ("Sexe", "Female"),
            ("Âge", "74 ans"),
            ("Date du test", "2024-07-17"),
            ("Localisation", "Halabja"),
            ("Méthode de détection", "Isolation Forest"),])

    # --------------------------------------------------------
    # 2. DESCRIPTION DE L'ANOMALIE
    # --------------------------------------------------------

    add_section("2. Description de l'anomalie")

    add_text("Le système LabGuard AI a signalé une valeur de glucose "
        "de 10 042 mg/dL comme statistiquement extrêmement atypique.")

    add_text("Cette valeur est très éloignée de l'intervalle de référence "
        "indiqué dans le dataset (70–99 mg/dL).")

    add_text("Dans le cadre de cette étude, cette observation est considérée "
        "comme une erreur d'enregistrement présumée, en raison de son "
        "caractère extrêmement inhabituel.")

    add_text("Une vérification de la donnée source serait nécessaire pour "
        "confirmer cette hypothèse.")

    # --------------------------------------------------------
    # 3. VÉRIFICATION INITIALE
    # --------------------------------------------------------

    add_section("3. Vérification initiale")

    add_table(
        ["Point à vérifier", "Résultat"],
        [
            (
                "Identification de l'échantillon",
                "Sample ID 23247 disponible dans le dataset"
            ),
            ("Nom de l'examen", "Glucose"),
            ("Unité", "mg/dL"),
            ("Valeur enregistrée", "10 042 mg/dL"),
            ("Intervalle de référence", "70–99 mg/dL"),
            ("Outcome", "Abnormal"),
            ("Erreur de saisie ou de transcription","Suspectée, non confirmée"),
            ("Donnée source originale", "À vérifier"),])

    # --------------------------------------------------------
    # 4. INVESTIGATION DES CAUSES POSSIBLES
    # --------------------------------------------------------

    add_section("4. Investigation des causes possibles")

    add_bold_text("Hypothèse principale : erreur d'enregistrement "
        "ou de transcription.")

    add_text("La valeur présente un caractère extrêmement inhabituel "
        "par rapport à l'intervalle de référence fourni.")

    add_text("Les vérifications recommandées sont :")

    checks = ["Comparer la valeur enregistrée avec le résultat original.",
        "Vérifier une éventuelle erreur de saisie ou de transcription.",
        "Vérifier les informations relatives à l'échantillon.",
        "Examiner les données analytiques disponibles, si elles existent."]

    for item in checks:
        doc.add_paragraph(item, style="List Bullet")

    p = doc.add_paragraph()
    p.add_run("Statut de la cause : ").bold = True
    p.add_run("hypothèse à confirmer.")

    # --------------------------------------------------------
    # 5. CONCLUSION DE L'INVESTIGATION
    # --------------------------------------------------------

    add_section("5. Conclusion de l'investigation")

    add_bold_text("Conclusion provisoire :")

    add_text("Valeur statistiquement extrêmement atypique, considérée "
        "comme une erreur d'enregistrement présumée dans le cadre "
        "du projet LabGuard AI.")

    add_text("La confirmation nécessite une vérification de la donnée "
        "source par une personne autorisée.")

    p = doc.add_paragraph()
    p.add_run("Statut : ").bold = True
    p.add_run("investigation en attente de confirmation.")

    # --------------------------------------------------------
    # 6. ACTIONS ET DÉCISION
    # --------------------------------------------------------

    add_section("6. Actions et décision")

    add_table(
        ["Action", "Statut"],
        [
            (
                "Signalement par LabGuard AI",
                "À réaliser lors du test du modèle"
            ),
            (
                "Vérification de la donnée source",
                "À réaliser"
            ),
            (
                "Confirmation de l'erreur éventuelle",
                "En attente"
            ),
            (
                "Correction de la donnée, si justifiée",
                "En attente de décision autorisée"
            ),])

    add_text("Aucune correction automatique de la valeur originale "
        "n'est effectuée par LabGuard AI.")

    # --------------------------------------------------------
    # 7. VÉRIFICATION ET CLÔTURE
    # --------------------------------------------------------

    add_section("7. Vérification et clôture")

    add_table(["Information", "Statut"],
        [("Cause confirmée", "Non"),
            ("Action corrective", "Non définie à ce stade"),
            ("Décision finale", "En attente"),
            ("Date de clôture", "Non clôturée"),])

    # --------------------------------------------------------
    # FINAL REMARK
    # --------------------------------------------------------

    doc.add_paragraph()

    p = doc.add_paragraph()
    p.add_run("Remarque : ").bold = True

    p.add_run("cet exemple est un cas d'étude construit à partir des "
        "données du dataset. Il ne constitue pas un dossier réel "
        "de non-conformité d'un laboratoire.")

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER

    footer.add_run("LabGuard AI – Fiche d'investigation | LAB-ANOM-0001")

    # --------------------------------------------------------
    # EXPORT TO MEMORY
    # --------------------------------------------------------

    output = BytesIO()

    doc.save(output)

    output.seek(0)

    return output


# ============================================================
# 9. GENERATE REPORT ENDPOINT
# ============================================================

@app.post("/generate-report")
def generate_report(request: AskRequest):

    # The endpoint is restricted to the specific case.
    question_lower = request.question.lower()

    if ("23247" not in question_lower
        or "glucose" not in question_lower):

        raise HTTPException(
            status_code=400,
            detail=("This report is only available for "
                "Glucose Sample ID 23247."))

    try:

        # Verify that FAISS retrieves the correct case.
        retrieved = faiss_search(
            request.question,
            top_k=request.top_k)

        glucose_case_found = any(
            doc.get("source") == "Cas_Glucose"
            and "23247" in doc.get("text", "")
            for doc in retrieved)

        if not glucose_case_found:

            raise HTTPException(
                status_code=404,
                detail=("The Glucose Sample ID 23247 document "
                    "was not retrieved by FAISS."))

        # Generate Word document.
        word_file = generate_glucose_investigation_form()

        return StreamingResponse(
            word_file,
            media_type=("application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"),
            headers={
                "Content-Disposition": (
                    'attachment; filename="LAB-ANOM-0001_Glucose_23247.docx"')})

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(f"Word generation error: "
                f"{type(e).__name__}: {e}"))
