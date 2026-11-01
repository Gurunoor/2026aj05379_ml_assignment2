import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef)

st.set_page_config(page_title="Tumour Classifier Lab", page_icon="🔬", layout="wide")

TARGET = "diagnosis"
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}


@st.cache_resource
def load_model(path):
    return joblib.load(path)


st.title("🔬 Breast Cancer Diagnosis – Model Comparison Lab")
st.caption("Pre-trained classifiers evaluated on an uploaded test CSV (no retraining happens here). "
           "Target column: `diagnosis` (1 = malignant, 0 = benign).")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV (features + diagnosis)", type="csv")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))

if uploaded is None:
    st.info("⬅️ Upload a test CSV in the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET not in df.columns:
    st.error(f"Uploaded file must contain the target column `{TARGET}`.")
    st.stop()

X, y = df.drop(columns=TARGET), df[TARGET]
model = load_model(MODEL_FILES[model_name])
try:
    X = X[list(model.feature_names_in_)]      # enforce training column order
except KeyError as e:
    st.error(f"Missing required feature columns: {e}")
    st.stop()

st.write(f"Loaded **{df.shape[0]} rows × {df.shape[1]} columns** – model selected: **{model_name}**")
st.dataframe(df.head(10), use_container_width=True)
