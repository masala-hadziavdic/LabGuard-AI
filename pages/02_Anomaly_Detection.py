from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION ET CHARGEMENT DES FICHIERS
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parents[1]

FILE_PREDICTIONS = PROJECT_PATH / "IsolationForest_All_Predictions.csv"
FILE_EXTREMES = PROJECT_PATH / "IsolationForest_Extreme_Anomalies.csv"


@st.cache_data
def load_data():
    predictions = pd.read_csv(FILE_PREDICTIONS, encoding="utf-8-sig")
    extremes = pd.read_csv(FILE_EXTREMES, encoding="utf-8-sig")
    return predictions, extremes


st.title("🔎 Anomaly Detection – Isolation Forest")

st.markdown("""
Cette page présente les résultats de détection des valeurs
statistiquement atypiques dans les données de laboratoire.

**Important :** une anomalie statistique constitue une alerte
pour investigation et ne confirme pas une erreur analytique.
""")

if not FILE_PREDICTIONS.exists() or not FILE_EXTREMES.exists():
    st.error(
        "Fichiers Isolation Forest introuvables. "
        "Vérifie que les deux fichiers CSV sont dans le dossier principal du projet.")
    st.stop()

df_all, df_extreme = load_data()

# ============================================================
# 2. NORMALISATION DES COLONNES
# ============================================================

def prepare_columns(df):
    df = df.copy()

    if "Test Name" in df.columns:
        df["_Test"] = df["Test Name"]
    elif "Test_Name" in df.columns:
        df["_Test"] = df["Test_Name"]
    else:
        df["_Test"] = "Unknown"

    if "Unit" in df.columns:
        df["_Unit"] = df["Unit"]
    elif "Model_Unit" in df.columns:
        df["_Unit"] = df["Model_Unit"]
    else:
        df["_Unit"] = "Unknown"

    if "Lab" in df.columns:
        df["_Lab"] = df["Lab"]
    else:
        df["_Lab"] = "Unknown"

    return df


df_all = prepare_columns(df_all)
df_extreme = prepare_columns(df_extreme)

df_all["IF_Prediction"] = pd.to_numeric(
    df_all["IF_Prediction"], errors="coerce")

df_anomalies = df_all[
    df_all["IF_Prediction"] == -1].copy()

# ============================================================
# 3. INDICATEURS PRINCIPAUX
# ============================================================

total_results = len(df_all)
total_anomalies = len(df_anomalies)

percentage_anomalies = (
    total_anomalies / total_results * 100
    if total_results > 0 else 0
)

total_extremes = len(df_extreme)

st.subheader("📊 Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Results analyzed",
    f"{total_results:,}".replace(",", " "))

col2.metric(
    "Anomalies detected",
    f"{total_anomalies:,}".replace(",", " "))

col3.metric(
    "Anomaly rate",
    f"{percentage_anomalies:.2f}%")

col4.metric(
    "Extreme values selected",
    f"{total_extremes:,}".replace(",", " "))

st.caption(
    "Les anomalies détectées correspondent à toutes les observations "
    "classées -1 par Isolation Forest. Les valeurs extrêmes sélectionnées "
    "constituent un sous-ensemble destiné à l'examen.")


# ============================================================
# 4. FILTRES
# ============================================================

st.subheader("🔍 Filters")

filter_col1, filter_col2, filter_col3 = st.columns(3)


def get_options(df, column):
    values = (
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist())
    return ["All"] + sorted(values)


with filter_col1:
    selected_test = st.selectbox(
        "Biological test",
        get_options(df_all, "_Test"))

with filter_col2:
    selected_unit = st.selectbox(
        "Unit",
        get_options(df_all, "_Unit"))

with filter_col3:
    selected_lab = st.selectbox(
        "Laboratory",
        get_options(df_all, "_Lab"))


def apply_filters(df):
    filtered = df.copy()

    if selected_test != "All":
        filtered = filtered[
            filtered["_Test"].astype(str) == selected_test]

    if selected_unit != "All":
        filtered = filtered[
            filtered["_Unit"].astype(str) == selected_unit]

    if selected_lab != "All":
        filtered = filtered[
            filtered["_Lab"].astype(str) == selected_lab]

    return filtered


filtered_anomalies = apply_filters(df_anomalies)
filtered_extremes = apply_filters(df_extreme)

# ============================================================
# 5. REPARTITION DES ANOMALIES PAR TEST
# ============================================================

st.subheader("📈 Anomalies by biological test")

if not filtered_anomalies.empty:

    TOP_N = 20

    anomalies_by_test = (
        filtered_anomalies
        .groupby("_Test")
        .size()
        .sort_values(ascending=False)
        .head(TOP_N)
        .sort_values(ascending=True))

    fig, ax = plt.subplots(figsize=(10, 8))

    anomalies_by_test.plot(
        kind="barh",
        ax=ax)

    ax.set_xlabel("Number of anomalies")
    ax.set_ylabel("Biological test")
    ax.set_title(
        f"Top {TOP_N} Biological Tests with Detected Anomalies")

    for i, value in enumerate(anomalies_by_test):
        ax.text(value, i, f" {value}", va="center")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("No anomalies found for the selected filters.")


# ============================================================
# 6. TABLEAU DES VALEURS EXTREMES
# ============================================================

st.subheader("🚨 Extreme values selected")

st.caption(
    "Les observations affichées sont les valeurs extrêmes "
    "sélectionnées parmi les anomalies détectées.")


def format_result(value):
    """Affiche les résultats avec un espace comme séparateur des milliers."""

    if pd.isna(value):
        return ""

    try:
        number = float(value)

        if number.is_integer():
            return f"{int(number):,}".replace(",", " ")

        return (
            f"{number:,.2f}"
            .replace(",", " ")
            .rstrip("0")
            .rstrip("."))

    except (ValueError, TypeError):
        return str(value)


def format_sample_id(value):
    """Conserve le Sample ID sans séparateur des milliers."""

    if pd.isna(value):
        return ""

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

        return str(value)

    except (ValueError, TypeError):
        return str(value)


if not filtered_extremes.empty:

    preferred_columns = [
        "Sample ID",
        "Lab",
        "Gender",
        "Date of Test Taken",
        "Age",
        "Test Name",
        "Unit",
        "Result",
        "Normal Range",
        "Outcome",
        "IF_Prediction",
        "Anomaly_Score",
        "Extreme_Type"]

    display_columns = [
        col for col in preferred_columns
        if col in filtered_extremes.columns]

    # Copie réservée à l'affichage
    df_display = filtered_extremes[
        display_columns].copy()

    # Formatage des identifiants
    if "Sample ID" in df_display.columns:
        df_display["Sample ID"] = (
            df_display["Sample ID"].apply(format_sample_id))

    # Formatage des résultats biologiques
    if "Result" in df_display.columns:
        df_display["Result"] = (
            df_display["Result"].apply(format_result))

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True)

else:
    st.info("No extreme values found for the selected filters.")


# ============================================================
# 7. TELECHARGEMENT DES RESULTATS
# ============================================================

st.subheader("⬇️ Download results")

col_download1, col_download2 = st.columns(2)

with col_download1:

    csv_anomalies = (
        filtered_anomalies
        .drop(columns=["_Test", "_Unit", "_Lab"], errors="ignore")
        .to_csv(index=False)
        .encode("utf-8-sig"))

    st.download_button(
        label="Download all detected anomalies",
        data=csv_anomalies,
        file_name="IsolationForest_Filtered_Anomalies.csv",
        mime="text/csv",
        use_container_width=True)

with col_download2:

    csv_extremes = (
        filtered_extremes
        .drop(columns=["_Test", "_Unit", "_Lab"], errors="ignore")
        .to_csv(index=False)
        .encode("utf-8-sig"))

    st.download_button(
        label="Download selected extreme values",
        data=csv_extremes,
        file_name="IsolationForest_Filtered_Extreme_Anomalies.csv",
        mime="text/csv",
        use_container_width=True)


# ============================================================
# 8. NOTE METHODOLOGIQUE
# ============================================================

st.divider()

st.markdown("""
**Methodological note**

Isolation Forest identifies statistically atypical observations.
The anomaly score is used as a model output for ranking observations.

The detected anomalies require further investigation and should not
be interpreted automatically as analytical errors or clinical abnormalities.
""")
