# Breast Cancer Diagnosis – ML Classification & Streamlit Deployment

## a. Problem Statement
Predict whether a breast tumour is **malignant (1)** or **benign (0)** from numeric measurements of cell nuclei
taken from digitised fine-needle-aspirate images. Five classifiers (Logistic Regression, Decision Tree, KNN,
Gaussian Naive Bayes, SVM) are trained on the same data, compared on six metrics, and served through an
interactive Streamlit app that evaluates the saved models on an uploaded test CSV.

## b. Dataset Description
- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset – UCI Machine Learning Repository (also bundled with scikit-learn).
- **Instances:** 569  (≥ 500 ✔)  |  **Features:** 30 numeric (≥ 12 ✔)  |  **Task:** binary classification
- **Features:** mean, standard error and "worst" value of 10 nuclear properties (radius, texture, perimeter, area,
  smoothness, compactness, concavity, concave points, symmetry, fractal dimension).
- **Target (`diagnosis`):** 1 = malignant (212 samples), 0 = benign (357 samples) – mildly imbalanced.
- **Split:** 75 % train / 25 % test (stratified, `random_state=2026`) → 426 train, 143 test. Test set saved as `test_data.csv`.
- **Preprocessing:** StandardScaler (inside a Pipeline) for Logistic Regression, KNN and SVM; none for tree / Naive Bayes.

## c. GitHub Repository Link
https://github.com/Gurunoor/2026aj05379_ml_assignment2

Live app: https://2026aj05379mlassignment2.streamlit.app/

## d. Models Used

### Comparison Table (evaluated on the 143-row held-out test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9650 | 0.9843 | 0.9800 | 0.9245 | 0.9515 | 0.9251 |
| Decision Tree | 0.9231 | 0.9401 | 0.9038 | 0.8868 | 0.8952 | 0.8346 |
| KNN | 0.9580 | 0.9802 | 0.9608 | 0.9245 | 0.9423 | 0.9098 |
| Naive Bayes | 0.9301 | 0.9881 | 0.9216 | 0.8868 | 0.9038 | 0.8493 |
| Support Vector Machine | 0.9790 | 0.9868 | 0.9808 | 0.9623 | 0.9714 | 0.9550 |

Precision / Recall / F1 are for the positive class (malignant).

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong (96.5 % accuracy, AUC 0.984) for a simple linear model, which indicates the classes are almost linearly separable after scaling. Highest-tier precision (0.98) but misses ~7.5 % of malignant cases (recall 0.925). Fast, interpretable baseline. |
| Decision Tree | Weakest overall (92.3 % accuracy, AUC 0.94, MCC 0.83). A single depth-limited tree is high-variance and its hard splits give coarse probability estimates, hurting AUC most. Easy to explain but not the best choice for accuracy here. |
| KNN | Good (95.8 % accuracy, MCC 0.91) once features are standardised; without scaling large-valued features such as area would dominate the distance. Recall equals Logistic Regression's. Slightly behind the SVM and slower at prediction time. |
| Naive Bayes | Accuracy is modest (93.0 %) because the 30 features are strongly correlated, violating the independence assumption, yet it has the **second-best AUC (0.988)** – its probability *ranking* is good while the 0.5 threshold is poorly calibrated, so recall is lower (0.887). |
| Support Vector Machine | Best on every threshold-based metric: 97.9 % accuracy, F1 0.971, MCC 0.955, and the highest recall (0.962). The RBF kernel captures mild non-linearity on scaled data while its margin maximisation generalises well on this small dataset. |
| **Overall winner** | **Support Vector Machine (RBF)** – highest Accuracy, Precision, Recall, F1 and MCC, with AUC (0.987) essentially tied with the best (Naive Bayes 0.988). Highest recall matters most clinically, since a missed malignancy is the costly error. Caveat: the test set is only 143 rows, so differences of 1–2 samples are within noise. |

## Repository Structure
```
├── app.py                 # Streamlit app
├── train_models.py        # trains & saves the 5 models, creates test_data.csv
├── requirements.txt
├── README.md
├── test_data.csv          # held-out test data (features + diagnosis)
├── full_dataset.csv
└── model/                 # *.joblib saved models + metrics_table.csv
```

## Streamlit App Features
CSV upload (test data) · model-selection dropdown · metric cards (Accuracy, AUC, Precision, Recall, F1, MCC) ·
confusion matrix, classification report and ROC curve tabs · optional side-by-side comparison of all five models.
Models are loaded from disk – **no retraining** on uploaded data.

## Run Locally
```bash
pip install -r requirements.txt
python train_models.py     # optional – regenerates models & test_data.csv
streamlit run app.py
```
