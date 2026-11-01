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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

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


def build_logistic_regression():
    # scaling inside the pipeline so it is applied automatically at prediction time
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, random_state=SEED))])


def build_decision_tree():
    # depth / leaf limits keep the tree from over-fitting
    return DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED)


def build_knn():
    # distance based -> needs scaled features
    return Pipeline([("sc", StandardScaler()),
                     ("clf", KNeighborsClassifier(n_neighbors=7))])


def build_naive_bayes():
    return GaussianNB()


if __name__ == "__main__":
    df = load_data()
    df.to_csv("full_dataset.csv", index=False)
    X_train, X_test, y_train, y_test = split_data(df)
    pd.concat([X_test, y_test], axis=1).to_csv("test_data.csv", index=False)

    rows = []
    rows.append(train_and_save("Logistic Regression", build_logistic_regression(), X_train, y_train, X_test, y_test))
    rows.append(train_and_save("Decision Tree", build_decision_tree(), X_train, y_train, X_test, y_test))
    rows.append(train_and_save("KNN", build_knn(), X_train, y_train, X_test, y_test))
    rows.append(train_and_save("Naive Bayes", build_naive_bayes(), X_train, y_train, X_test, y_test))

    res = pd.DataFrame(rows).round(4)
    res.to_csv("model/metrics_table.csv", index=False)
    print(res.to_string(index=False))
    print("\nShape:", df.shape, "| train:", X_train.shape, "test:", X_test.shape)
