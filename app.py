import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             classification_report, roc_curve)

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

def score(m):
    pred = m.predict(X)
    proba = m.predict_proba(X)[:, 1]
    return pred, proba, {
        "Accuracy": accuracy_score(y, pred), "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred), "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred), "MCC": matthews_corrcoef(y, pred)}

pred, proba, metrics = score(model)

with st.expander("Preview uploaded data", expanded=False):
    st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True)

st.subheader(f"Results – {model_name}")
cols = st.columns(6)
for c, (k, v) in zip(cols, metrics.items()):
    c.metric(k, f"{v:.4f}")

tab1, tab2, tab3 = st.tabs(["Confusion matrix", "Classification report", "ROC curve"])
with tab1:
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Purples", cbar=False,
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with tab2:
    rep = pd.DataFrame(classification_report(y, pred, target_names=["Benign", "Malignant"],
                                             output_dict=True)).T.round(3)
    st.dataframe(rep, use_container_width=True)
with tab3:
    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend()
    st.pyplot(fig)
