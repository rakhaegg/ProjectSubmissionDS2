import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# ---------------------------------------------------------
# Load trained pipeline (preprocess + LogisticRegression)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model ...")
def load_model():
    return joblib.load("dropout_model.pkl")

model = load_model()
EXPECTED_COLS = list(model.named_steps["prep"].feature_names_in_)

# ---------------------------------------------------------
# UI Sidebar
# ---------------------------------------------------------
st.title("Early‑Warning Dropout Predictor")
st.write("Masukkan informasi akademik dan finansial mahasiswa untuk memprediksi risiko dropout.")

st.sidebar.header("Input Mahasiswa")
age          = st.sidebar.number_input("Usia saat mendaftar (tahun)", 17, 70, 20, key="age")
adm_grade     = st.sidebar.number_input("Admission grade (0‑200)", 0.0, 200.0, 130.0, step=0.1, key="adm_grade")

st.sidebar.markdown("### Semester 1")
sem1_enrolled = st.sidebar.number_input("Mata kuliah diambil S1", 0, 40, 6, key="sem1_enrolled")
sem1_approved = st.sidebar.number_input("Mata kuliah lulus S1", 0, 40, 5, key="sem1_approved")

st.sidebar.markdown("### Semester 2")
sem2_enrolled = st.sidebar.number_input("Mata kuliah diambil S2", 0, 40, 6, key="sem2_enrolled")
sem2_approved = st.sidebar.number_input("Mata kuliah lulus S2", 0, 40, 4, key="sem2_approved")

st.sidebar.markdown("### Status Keuangan")
late_fee = st.sidebar.selectbox("Apakah telat bayar semester berjalan?", ("Tidak", "Ya"), key="late_fee")
debtor   = st.sidebar.selectbox("Apakah tercatat Debtor?", ("Tidak", "Ya"), key="debtor")

# ---------------------------------------------------------
# Helper untuk membangun DataFrame fitur lengkap
# ---------------------------------------------------------

def build_feature_df():
    df = pd.DataFrame([{
        "Age_at_enrollment": age,
        "Admission_grade": adm_grade,
        "Curricular_units_1st_sem_enrolled": sem1_enrolled,
        "Curricular_units_1st_sem_approved": sem1_approved,
        "Curricular_units_2nd_sem_enrolled": sem2_enrolled,
        "Curricular_units_2nd_sem_approved": sem2_approved,
        "Tuition_fees_up_to_date": 0 if late_fee == "Tidak" else 1,
        "Debtor": 0 if debtor == "Tidak" else 1,
    }])

    df["pass_rate_sem1"] = (df["Curricular_units_1st_sem_approved"] / df["Curricular_units_1st_sem_enrolled"].replace(0, np.nan)).fillna(0)
    df["pass_rate_sem2"] = (df["Curricular_units_2nd_sem_approved"] / df["Curricular_units_2nd_sem_enrolled"].replace(0, np.nan)).fillna(0)
    df["finance_risk"]   = ((df["Tuition_fees_up_to_date"] == 1) | (df["Debtor"] == 1)).astype(int)

    # Tambahkan kolom lain sebagai NaN lalu reindex agar urut & komplit
    df_full = df.reindex(columns=EXPECTED_COLS, fill_value=np.nan)
    return df_full

# ---------------------------------------------------------
# Predict & Explain
# ---------------------------------------------------------
if st.button("Prediksi Risiko Dropout"):
    feat_df = build_feature_df()

    prob = model.predict_proba(feat_df)[0, 1]
    risk_level = "No‑risk" if prob < 0.40 else ("Yellow" if prob < 0.65 else "Red")

    st.metric("Probabilitas Dropout", f"{prob:.1%}")
    st.write(f"**Kategori:** {risk_level}")

    st.subheader("Kontributor Risiko (Top‑5)")

    # SHAP linear explainer
    X_trans = model.named_steps["prep"].transform(feat_df)
    feature_names = model.named_steps["prep"].get_feature_names_out()

    explainer = shap.LinearExplainer(model.named_steps["clf"], X_trans, feature_names=feature_names, model_output="log_odds")
    shap_vals = explainer.shap_values(X_trans)[0]

    top_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]
    top_df  = pd.DataFrame({
        "Feature": feature_names[top_idx],
        "Impact (± log‑odds)": shap_vals[top_idx]
    })

    st.dataframe(top_df, use_container_width=True)
    st.caption("Positif ⇒ menaikkan risiko, Negatif ⇒ menurunkan risiko.")
