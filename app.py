import os
import pickle
from pathlib import Path
from io import BytesIO

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import faiss

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_mistralai import ChatMistralAI

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LabGuard AI",
    page_icon="🧪",
    layout="wide")

st.title("🧪 LabGuard AI")
st.caption("Laboratory Data Analysis & Quality Document Assistant")


# ============================================================
# 2. PATHS AND API KEY
# ============================================================

BASE_PATH = Path(__file__).resolve().parent

DATA_PATH = BASE_PATH / "Laboratory_Data_Final_Merged_Clean.csv"

RAG_PATH = BASE_PATH / "rag"

FAISS_PATH = RAG_PATH / "faiss_index.bin"

METADATA_PATH = RAG_PATH / "metadata.pkl"

load_dotenv(BASE_PATH / ".env")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


# ============================================================
# 3. LOAD LABORATORY DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH)

# ============================================================
# 4. LOAD RAG COMPONENTS
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource
def load_faiss_index():

    if not FAISS_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_PATH}")

    return faiss.read_index(str(FAISS_PATH))


@st.cache_resource
def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}")

    with open(METADATA_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_mistral():

    if not MISTRAL_API_KEY:
        raise ValueError(
            "MISTRAL_API_KEY is missing. "
            "Check your .env file.")

    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        max_retries=2)

# ============================================================
# 5. RAG RETRIEVAL: FAISS SEARCH
# ============================================================

def faiss_search(question, top_k=3):

    embedding_model = load_embedding_model()
    faiss_index = load_faiss_index()
    metadata = load_metadata()

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

        result = metadata[idx].copy()

        result["similarity_score"] = float(score)

        results.append(result)

    return results

# ============================================================
# 6. RAG GENERATION: MISTRAL
# ============================================================

def generate_rag_answer(question, contexts):

    mistral_llm = load_mistral()

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are LabGuard AI, a laboratory quality document assistant.

Answer the question using ONLY the information
in the retrieved documents.

Rules:
- Do not invent information.
- Do not use external knowledge.
- If the documents do not contain enough information,
  clearly say that the information is not available.
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

    response = mistral_llm.invoke(prompt)

    return response.content

# ============================================================
# 7. COMPLETE RAG FUNCTION
# ============================================================

def ask_labguard(question, top_k=3):

    retrieved = faiss_search(question,top_k=top_k)

    if not retrieved:
        return ("No relevant document was retrieved.",[])

    contexts = [
        (
            f"[Source: {r['source']} | "
            f"Chunk: {r['chunk_id']}]\n"
            f"{r['text']}")
        for r in retrieved]

    answer = generate_rag_answer(question,contexts)

    return answer, retrieved


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

        table = doc.add_table(
            rows=1,
            cols=len(headers))

        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        # Header row
        for i, header in enumerate(headers):

            cell = table.rows[0].cells[i]
            cell.text = header

            shading = cell._tc.get_or_add_tcPr()

            shading.append(
                parse_xml(
                    r'<w:shd {} w:fill="1F4E79"/>'.format(
                        nsdecls("w"))))

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

                cells[i].vertical_alignment = (
                    WD_CELL_VERTICAL_ALIGNMENT.CENTER)

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
        run = p.add_run(text)
        run.bold = True

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    add_title("CAS D'ÉTUDE GLUCOSE")

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    run = subtitle.add_run(
        "CAS D'ÉTUDE – DONNÉES DU DATASET")

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

    add_text(
        "Le système LabGuard AI a signalé une valeur de glucose "
        "de 10 042 mg/dL comme statistiquement extrêmement atypique.")

    add_text(
        "Cette valeur est très éloignée de l'intervalle de référence "
        "indiqué dans le dataset (70–99 mg/dL).")

    add_text(
        "Dans le cadre de cette étude, cette observation est considérée "
        "comme une erreur d'enregistrement présumée, en raison de son "
        "caractère extrêmement inhabituel.")

    add_text(
        "Une vérification de la donnée source serait nécessaire pour "
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
            (
                "Erreur de saisie ou de transcription",
                "Suspectée, non confirmée"
            ),
            ("Donnée source originale", "À vérifier"),])

    # --------------------------------------------------------
    # 4. INVESTIGATION DES CAUSES POSSIBLES
    # --------------------------------------------------------

    add_section("4. Investigation des causes possibles")

    add_bold_text(
        "Hypothèse principale : erreur d'enregistrement "
        "ou de transcription.")

    add_text(
        "La valeur présente un caractère extrêmement inhabituel "
        "par rapport à l'intervalle de référence fourni.")

    add_text("Les vérifications recommandées sont :")

    checks = [
        "Comparer la valeur enregistrée avec le résultat original.",
        "Vérifier une éventuelle erreur de saisie ou de transcription.",
        "Vérifier les informations relatives à l'échantillon.",
        "Examiner les données analytiques disponibles, si elles existent.",]

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

    add_text(
        "La confirmation nécessite une vérification de la donnée "
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
                "En attente de décision autorisée"),])

    add_text(
        "Aucune correction automatique de la valeur originale "
        "n'est effectuée par LabGuard AI.")

    # --------------------------------------------------------
    # 7. VÉRIFICATION ET CLÔTURE
    # --------------------------------------------------------

    add_section("7. Vérification et clôture")

    add_table(
        ["Information", "Statut"],
        [
            ("Cause confirmée", "Non"),
            ("Action corrective", "Non définie à ce stade"),
            ("Décision finale", "En attente"),
            ("Date de clôture", "Non clôturée"),])

    # --------------------------------------------------------
    # FINAL REMARK
    # --------------------------------------------------------

    doc.add_paragraph()

    p = doc.add_paragraph()

    p.add_run("Remarque : ").bold = True

    p.add_run(
        "cet exemple est un cas d'étude construit à partir des "
        "données du dataset. Il ne constitue pas un dossier réel "
        "de non-conformité d'un laboratoire.")

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    footer = section.footer.paragraphs[0]

    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER

    footer.add_run(
        "LabGuard AI – Fiche d'investigation | LAB-ANOM-0001")

    # --------------------------------------------------------
    # EXPORT TO MEMORY
    # --------------------------------------------------------

    output = BytesIO()

    doc.save(output)

    output.seek(0)

    return output


# ============================================================
# 9. LOAD DATA
# ============================================================

try:

    df = load_data()

except Exception as e:

    st.error(f"Dataset loading error: {e}")
    st.stop()


# ============================================================
# 10. APPLICATION TABS
# ============================================================

tab1, tab2 = st.tabs([
    "📊 Laboratory Dashboard",
    "🔬 RAG Document Assistant"])


# ============================================================
# TAB 1: LABORATORY DASHBOARD
# ============================================================

with tab1:

    st.header("Laboratory Data Dashboard")

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.sidebar.header("Data Filters")

    labs = (
        ["All"]
        + sorted(df["Lab"].dropna().unique().tolist()))

    selected_lab = st.sidebar.selectbox(
        "Laboratory",
        labs)

    tests = (
        ["All"]
        + sorted(df["Test Name"].dropna().unique().tolist()))

    selected_test = st.sidebar.selectbox(
        "Test",
        tests)

    filtered_df = df.copy()

    if selected_lab != "All":

        filtered_df = filtered_df[
            filtered_df["Lab"] == selected_lab]

    if selected_test != "All":

        filtered_df = filtered_df[
            filtered_df["Test Name"] == selected_test]

    # --------------------------------------------------------
    # KPI INDICATORS
    # --------------------------------------------------------

    total_results = len(filtered_df)

    outcome_clean = (
        filtered_df["Outcome"]
        .astype(str)
        .str.strip()
        .str.lower())

    abnormal_count = (
        outcome_clean == "abnormal").sum()

    normal_count = (
        outcome_clean == "normal").sum()

    abnormal_rate = (
        abnormal_count / total_results * 100
        if total_results > 0 else 0)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Results",
        f"{total_results:,}")

    col2.metric(
        "Abnormal Results",
        f"{abnormal_count:,}")

    col3.metric(
        "Normal Results",
        f"{normal_count:,}")

    col4.metric(
        "Abnormal Rate",
        f"{abnormal_rate:.1f}%")

    st.divider()

    # --------------------------------------------------------
    # NORMAL VS ABNORMAL
    # --------------------------------------------------------

    st.subheader("Outcome Distribution")

    outcome_counts = pd.Series({
        "Normal": normal_count,
        "Abnormal": abnormal_count})

    fig, ax = plt.subplots(figsize=(7, 4))

    outcome_counts.plot(
        kind="bar",
        ax=ax,
        color=["#269B83", "#D94B4B"],
        edgecolor="white")

    ax.set_ylabel("Number of Results")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # RESULTS BY LABORATORY
    # --------------------------------------------------------

    st.subheader("Results by Laboratory")

    lab_counts = filtered_df["Lab"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 4))

    lab_counts.plot(
        kind="bar",
        ax=ax,
        color="#3977A8")

    ax.set_ylabel("Number of Results")
    ax.set_xlabel("Laboratory")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # TOP TESTS WITH ABNORMAL RESULTS
    # --------------------------------------------------------

    st.subheader("Top 10 Tests with Abnormal Results")

    abnormal_df = filtered_df[
        outcome_clean == "abnormal"]

    top_tests = (
        abnormal_df["Test Name"]
        .value_counts()
        .head(10))

    fig, ax = plt.subplots(figsize=(10, 5))

    top_tests.sort_values().plot(
        kind="barh",
        ax=ax,
        color="#D94B4B")

    ax.set_xlabel("Number of Abnormal Results")
    ax.set_ylabel("Test Name")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    with st.expander("View filtered laboratory data"):

        st.dataframe(
            filtered_df,
            use_container_width=True)

    st.caption(
        "LabGuard AI | Laboratory Data Analysis")


# ============================================================
# TAB 2: RAG DOCUMENT ASSISTANT
# ============================================================

with tab2:

    st.header("🔬 LabGuard AI – Document Assistant")

    st.write(
        "Ask questions about laboratory procedures, "
        "anomaly investigation and quality documentation.")

    st.info(
        "The assistant uses the indexed LabGuard AI documents "
        "to retrieve relevant information and generate answers.")

    # --------------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------------

    question = st.text_area(
        "Enter your question:",
        placeholder=(
            "Example: Analyse le cas Glucose Sample ID 23247."),
        height=100)

    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    top_k = st.slider(
        "Number of retrieved document chunks",
        min_value=1,
        max_value=10,
        value=3)

    # --------------------------------------------------------
    # ASK BUTTON
    # --------------------------------------------------------

    if st.button(
        "Ask LabGuard AI",
        type="primary"):

        if not question.strip():

            st.warning(
                "Please enter a question first.")

        else:

            with st.spinner(
                "Searching documents and generating answer..."):

                try:

                    answer, retrieved = ask_labguard(
                        question,
                        top_k=top_k)

                    # ----------------------------------------
                    # DISPLAY ANSWER
                    # ----------------------------------------

                    st.subheader("Answer")

                    st.markdown(answer)

                    # ----------------------------------------
                    # DISPLAY SOURCES
                    # ----------------------------------------

                    st.subheader("Retrieved Sources")

                    for i, result in enumerate(
                        retrieved,
                        start=1):

                        title = (
                            f"Source {i}: "
                            f"{result['source']} "
                            f"(Chunk {result['chunk_id']})")

                        with st.expander(title):

                            st.write(
                                "Similarity score:",
                                round(
                                    result["similarity_score"],4))

                            st.markdown(result["text"])
                    # ----------------------------------------
                    # CHECK SPECIFIC GLUCOSE CASE
                    # ----------------------------------------

                    question_lower = question.lower()

                    # Both terms must appear in the question.
                    glucose_case_requested = (
                        "23247" in question_lower
                        and "glucose" in question_lower)

                    # Check whether FAISS retrieved the
                    # specific Cas_Glucose source recorded in metadata.
                    retrieved_glucose_case = any(

                        result.get("source") == "Cas_Glucose"

                        and "23247" in result.get("text", "")

                        for result in retrieved)

                    # ----------------------------------------
                    # WORD DOWNLOAD
                    # ----------------------------------------

                    if (glucose_case_requested
                        and retrieved_glucose_case):

                        st.markdown("---")

                        st.subheader("📄 Fiche d'investigation")

                        st.success("Le cas Glucose Sample ID 23247 "
                            "a été identifié dans les documents.")

                        word_file = (generate_glucose_investigation_form())

                        st.download_button(
                            label=("⬇️ Télécharger la fiche "
                                "Word préremplie"),

                            data=word_file,

                            file_name=("LAB-ANOM-0001_Glucose_23247.docx"),

                            mime=("application/vnd.openxmlformats-officedocument."
                                "wordprocessingml.document"),

                            key="download_glucose_23247")

                    else:

                        st.caption("Aucune fiche Word spécifique "
                            "n'est disponible pour cette question.")

                except Exception as e:

                    st.error(f"RAG error: {type(e).__name__}: {e}")

    st.divider()

    st.caption("LabGuard AI | RAG Document Assistant")
