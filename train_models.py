"""Train classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset,
save models + test data + metrics table."""
import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

SEED = 2026
TARGET = "diagnosis"   # 1 = malignant, 0 = benign


def load_data():
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df[TARGET] = 1 - df["target"]          # make 1 = malignant (positive class)
    df = df.drop(columns="target")
    return df


def split_data(df):
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)


def evaluate(name, model, X_test, y_test):
    """Return the six required metrics for one fitted model."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    }


def train_and_save(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, f"model/{name.lower().replace(' ', '_')}.joblib")
    return evaluate(name, model, X_test, y_test)


if __name__ == "__main__":
    df = load_data()
    df.to_csv("full_dataset.csv", index=False)
    X_train, X_test, y_train, y_test = split_data(df)
    pd.concat([X_test, y_test], axis=1).to_csv("test_data.csv", index=False)

    rows = []
    print('Data saved:', df.shape, X_train.shape, X_test.shape)
