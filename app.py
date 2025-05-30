import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.set_page_config(page_title="Prediksi Risiko Dropout", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("dropout_model.pkl")

model = load_model()
prep  = model.named_steps["prep"]
clf   = model.named_steps["clf"]

EXPECTED_INPUT_COLS = prep.feature_names_in_
FEATURE_NAMES        = prep.get_feature_names_out()

def add_engineered_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pass_rate_sem1"] = (
        df["Curricular_units_1st_sem_approved"]
        / df["Curricular_units_1st_sem_enrolled"].replace(0, np.nan)
    ).fillna(0)
    df["pass_rate_sem2"] = (
        df["Curricular_units_2nd_sem_approved"]
        / df["Curricular_units_2nd_sem_enrolled"].replace(0, np.nan)
    ).fillna(0)
    df["finance_risk"] = (
        (df["Tuition_fees_up_to_date"] == 0) | (df["Debtor"] == 1)
    ).astype(int)
    return df

@st.cache_data
def load_data_and_options():
    df = pd.read_csv("data.csv", sep=";")
    opts = {
        "Application_mode": sorted(df["Application_mode"].unique()),
        "Course"          : sorted(df["Course"].unique()),
        "Mothers_qualification": sorted(df["Mothers_qualification"].unique()),
        "Fathers_qualification": sorted(df["Fathers_qualification"].unique())
    }
    feat_df = add_engineered_cols(df)
    X_trans = prep.transform(feat_df)
    idx = np.random.choice(X_trans.shape[0], size=min(100, X_trans.shape[0]), replace=False)
    background = X_trans[idx]
    return opts, background

opts, BACKGROUND = load_data_and_options()

explainer = shap.Explainer(clf, BACKGROUND, feature_names=FEATURE_NAMES)


def build_feature_df(inputs: dict) -> pd.DataFrame:
    """Buat DataFrame 1 baris, engineered + reindex sesuai EXPECTED_INPUT_COLS."""
    df = pd.DataFrame([inputs])
    df = add_engineered_cols(df)
    df = df.reindex(columns=EXPECTED_INPUT_COLS, fill_value=0)
    return df

st.sidebar.header("📝 Input Data Mahasiswa")
admission_grade = st.sidebar.number_input(
    "Admission Grade", 95.0, 190.0, 127.0, step=0.1, key="adg"
)
age = st.sidebar.number_input(
    "Age at Enrollment", 17, 70, 23, step=1, key="age"
)
sem1_enrolled = st.sidebar.number_input(
    "1st Sem Enrolled", 0, 26, 6, step=1, key="s1e"
)
sem1_approved = st.sidebar.number_input(
    "1st Sem Approved", 0, 26, 5, step=1, key="s1a"
)
sem2_enrolled = st.sidebar.number_input(
    "2nd Sem Enrolled", 0, 26, 6, step=1, key="s2e"
)
sem2_approved = st.sidebar.number_input(
    "2nd Sem Approved", 0, 26, 5, step=1, key="s2a"
)
tuition_up = st.sidebar.selectbox(
    "Tuition Fees Up to Date?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No", key="tut"
)
debtor = st.sidebar.selectbox(
    "Debtor?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", key="deb"
)
gender = st.sidebar.selectbox(
    "Gender (0=Female,1=Male)", [0,1], key="gen"
)
scholar = st.sidebar.selectbox(
    "Scholarship Holder?", [0,1], format_func=lambda x: "No" if x==0 else "Yes", key="sch"
)
day_eve = st.sidebar.selectbox(
    "Daytime Attendance? (1=Yes)", [0,1], key="de"
)
intl = st.sidebar.selectbox(
    "International Student?", [0,1], key="intl"
)
edu_need = st.sidebar.selectbox(
    "Special Educational Needs?", [0,1], key="spn"
)
app_mode = st.sidebar.selectbox(
    "Application Mode", opts["Application_mode"], key="am"
)
course = st.sidebar.selectbox(
    "Course Code", opts["Course"], key="crs"
)
mom_qual = st.sidebar.selectbox(
    "Mother's Qualification", opts["Mothers_qualification"], key="mq"
)
dad_qual = st.sidebar.selectbox(
    "Father's Qualification", opts["Fathers_qualification"], key="dq"
)

if st.sidebar.button("Prediksi Risiko Dropout", key="pred"):
    user_inputs = {
        "Admission_grade": admission_grade,
        "Age_at_enrollment": age,
        "Curricular_units_1st_sem_enrolled": sem1_enrolled,
        "Curricular_units_1st_sem_approved": sem1_approved,
        "Curricular_units_2nd_sem_enrolled": sem2_enrolled,
        "Curricular_units_2nd_sem_approved": sem2_approved,
        "Tuition_fees_up_to_date": tuition_up,
        "Debtor": debtor,
        "Gender": gender,
        "Scholarship_holder": scholar,
        "Daytime_evening_attendance": day_eve,
        "International": intl,
        "Educational_special_needs": edu_need,
        "Application_mode": app_mode,
        "Course": course,
        "Mothers_qualification": mom_qual,
        "Fathers_qualification": dad_qual
    }

    feat_df = build_feature_df(user_inputs)

    score = model.predict_proba(feat_df)[:,1][0]
    level = "Red" if score>=0.65 else "Yellow" if score>=0.40 else "No-risk"

    st.metric("🔎 Probabilitas Dropout", f"{score:.1%}", delta=None)
    st.markdown(f"**Risk Level:** {level}")

    X_user = prep.transform(feat_df)
    shap_vals = explainer(X_user).values[0]  # array shape (n_features,)
    idx_top5 = np.argsort(np.abs(shap_vals))[-5:][::-1]
    top5 = pd.DataFrame({
        "Feature": FEATURE_NAMES[idx_top5],
        "Impact (± log-odds)": shap_vals[idx_top5]
    })

    st.subheader("Kontributor Risiko (Top-5)")
    st.dataframe(top5, use_container_width=True)
