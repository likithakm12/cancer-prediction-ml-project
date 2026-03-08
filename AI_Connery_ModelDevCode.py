# ============================================
# IMPORTS
# ============================================

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from imblearn.over_sampling import SMOTE


# ============================================
# PATH SETUP
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================
# 1. FEATURE ENGINEERING & TARGET PREPARATION
# ============================================

def feature_engineering(df):

    # Target
    df["Cancer"] = df["Cancer"].map({"YES": 1, "NO": 0})

    # Drop identifiers
    df.drop(["Patient_ID", "Patient_Name"], axis=1, inplace=True)

    # Tumor size category
    df["Tumor_Size_Category"] = pd.cut(
        df["Tumor_Size_cm"],
        bins=[0, 2, 4, 10],
        labels=["Low", "Medium", "High"]
    )

    # Age group
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[18, 30, 50, 120],
        labels=["Young", "Middle", "Old"]
    )

    # Lifestyle risk
    df["Lifestyle_Risk"] = df["Smoking"] + df["Alcohol"]

    # Clinical risk
    df["Clinical_Risk"] = df["Genetic_Marker"] + df["Family_History"]

    # BMI risk
    df["BMI_Risk"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 60],
        labels=["Underweight", "Normal", "Overweight"]
    )

    X = df.drop("Cancer", axis=1)
    y = df["Cancer"]

    print("\nAfter Feature Engineering:")
    print(y.value_counts().to_string())

    return X, y


# ============================================
# 2. ENCODING (LABEL + ONE-HOT)
# ============================================

def encode_features(X):

    # Label encoding (binary)
    X["Gender"] = X["Gender"].map({"Male": 1, "Female": 0})
    X["Region"] = X["Region"].map({"Urban": 1, "Rural": 0})

    onehot_cols = [
        "Ethnicity",
        "Tumor_Size_Category",
        "Age_Group",
        "BMI_Risk"
    ]

    numeric_cols = [c for c in X.columns if c not in onehot_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), onehot_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    return preprocessor


# ============================================
# 3. SMOTE
# ============================================

def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print("\nAfter SMOTE:")
    print("Cancer NO:", (y_res == 0).sum())
    print("Cancer YES:", (y_res == 1).sum())

    return X_res, y_res

# ============================================
# 4. SCALING
# ============================================

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


# ============================================
# 5. SPLIT
# ============================================

def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )


# ============================================
# 6. MODELS
# ============================================

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }


# ============================================
# 7. TRAIN,EVALUATE & CROSS VALIDATION
# ============================================

def train_and_evaluate(models, X_train, X_test, y_train, y_test):

    results = []

    for name, model in models.items():

        # Train
        model.fit(X_train, y_train)

        # Cross-Validation (5-fold ROC-AUC)
        cv_auc = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="roc_auc"
        ).mean()

        # Test predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Model_Object": model,
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_prob),
            "CV_ROC_AUC": cv_auc,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Confusion_Matrix": confusion_matrix(y_test, y_pred)
        })

    return results

# ============================================
# 8. PLOTTING
# ============================================

def plot_results(results):

    df = pd.DataFrame(results)

    # -------- Metric Bar Plots --------
    metrics = ["Recall", "F1", "ROC_AUC", "Accuracy", "Precision"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=df,
            x="Model",
            y=metric
        )
        plt.xticks(rotation=45)
        plt.title(f"{metric} Comparison Across Models")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{metric}.png"))
        plt.close()

    # -------- Confusion Matrix Plots --------
    for _, row in df.iterrows():
        cm = row["Confusion_Matrix"]
        model_name = row["Model"]

        plt.figure(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["NO", "YES"],
            yticklabels=["NO", "YES"]
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOT_DIR, f"Confusion_Matrix_{model_name}.png")
        )
        plt.close()

# ============================================
# 9. BEST MODEL SELECTION
# ============================================

def select_best_model(results):
    return sorted(
        results,
        key=lambda x: (x["Recall"], x["F1"], x["ROC_AUC"]),
        reverse=True
    )[0]


# ============================================
# 10. INFERENCE FUNCTIONS
# ============================================

def predict_single(model, preprocessor, scaler, input_row):
     """
    input_row: dict or pandas Series (single patient)
    """
     df = pd.DataFrame([input_row])
     X_enc = preprocessor.transform(df)
     X_scaled = scaler.transform(X_enc)
     return model.predict(X_scaled)[0]


def predict_batch(model, preprocessor, scaler, input_df):
    X_enc = preprocessor.transform(input_df)
    X_scaled = scaler.transform(X_enc)
    return model.predict(X_scaled)

# ============================================
# 11. SAVE MODEL METRICS (EXCEL)
# ============================================

from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage

def save_model_metrics(results, best_model):

    metrics_path = os.path.join(
        OUTPUT_DIR, "AI_Connery_ModelMetrics.xlsx"
    )

    # -------- All Models Metrics --------
    metrics_df = pd.DataFrame([
        {
            "Model": r["Model"],
            "Accuracy": r["Accuracy"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1": r["F1"],
            "ROC_AUC": r["ROC_AUC"],
            "CV_ROC_AUC": r["CV_ROC_AUC"]
        }
        for r in results
    ])

    # -------- Best Model --------
    best_model_df = pd.DataFrame([
        {
            "Selection": "Best Recall (Primary) & F1 (Secondary)",
            "Model": best_model["Model"],
            "Recall": best_model["Recall"],
            "F1": best_model["F1"],
            "Accuracy": best_model["Accuracy"],
            "Precision": best_model["Precision"],
            "ROC_AUC": best_model["ROC_AUC"]
        }
    ])

    # -------- Confusion Matrices (table) --------
    cm_rows = []
    for r in results:
        tn, fp, fn, tp = r["Confusion_Matrix"].ravel()
        cm_rows.append({
            "Model": r["Model"],
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

    cm_df = pd.DataFrame(cm_rows)

    # -------- Save tables first --------
    with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="All_Model_Metrics", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion_Matrices", index=False)
        best_model_df.to_excel(writer, sheet_name="Best_Model", index=False)
        
    # ---------- Observations Sheet (For Manual Summary) ----------
        pd.DataFrame(
            {
                "Observations": [
                    ""  
                ]
            }
        ).to_excel(writer, sheet_name="Observations", index=False)

    # -------- Insert plots as images --------
    wb = load_workbook(metrics_path)

    all_models_sheet = wb["All_Model_Metrics"]
    confusion_sheet = wb["Confusion_Matrices"]
    best_model_sheet = wb["Best_Model"]

    # -------- All models metric comparison plots --------
    all_plots = [
        "Accuracy.png",
        "Precision.png",
        "Recall.png",
        "F1.png",
        "ROC_AUC.png"
    ]

    row_pos = metrics_df.shape[0] + 3

    for plot in all_plots:
        img_path = os.path.join(PLOT_DIR, plot)
        if os.path.exists(img_path):
            img = XLImage(img_path)
            img.anchor = f"A{row_pos}"
            all_models_sheet.add_image(img)
            row_pos += 20

    # -------- Confusion matrix images for ALL models --------
    row_pos = cm_df.shape[0] + 3

    for r in results:
        model_name = r["Model"]
        img_path = os.path.join(
            PLOT_DIR, f"Confusion_Matrix_{model_name}.png"
        )

        if os.path.exists(img_path):
            img = XLImage(img_path)
            img.anchor = f"A{row_pos}"
            confusion_sheet.add_image(img)
            row_pos += 20

    # -------- Best model confusion matrix --------
    img_path = os.path.join(
        PLOT_DIR, f"Confusion_Matrix_{best_model['Model']}.png"
    )

    if os.path.exists(img_path):
        img = XLImage(img_path)
        img.anchor = "A5"
        best_model_sheet.add_image(img)

    wb.save(metrics_path)

    print("\nModel metrics & plots saved at:")
    print(metrics_path)


# ============================================
# MAIN EXECUTION
# ============================================

def main():

    # ---------- Load dataset ----------
    df = pd.read_excel(
        os.path.join(DATA_DIR, "Cancer_Patient_Dataset.xlsx"),
        sheet_name="Cleaned_Final_Data"
    )

    # ---------- Feature engineering ----------
    X, y = feature_engineering(df)

    # ---------- Encoding ----------
    preprocessor = encode_features(X)
    X_encoded = preprocessor.fit_transform(X)

    # ---------- Scaling ----------
    X_scaled, scaler = scale_features(X_encoded)

    # ---------- Handle imbalance ----------
    X_bal, y_bal = apply_smote(X_scaled, y)

    # ---------- Train-test split ----------
    X_train, X_test, y_train, y_test = split_data(X_bal, y_bal)

    # ============================================
    # SAVE PROCESSED DATASETS 
    # ============================================

    processed_path = os.path.join(
        OUTPUT_DIR, "AI_Connery_ProcessedDataset.xlsx"
    )

    feature_names = preprocessor.get_feature_names_out()

    if os.path.exists(processed_path):
        os.remove(processed_path)

    with pd.ExcelWriter(processed_path, engine="openpyxl") as writer:

        pd.DataFrame(
            X_train, columns=feature_names
        ).to_excel(writer, sheet_name="X_Train", index=False)

        pd.DataFrame(
            X_test, columns=feature_names
        ).to_excel(writer, sheet_name="X_Test", index=False)

        pd.DataFrame(
            y_train, columns=["Cancer"]
        ).to_excel(writer, sheet_name="y_Train", index=False)

        pd.DataFrame(
            y_test, columns=["Cancer"]
        ).to_excel(writer, sheet_name="y_Test", index=False)

    # ---------- Observations Sheet (For Manual Summary) ----------
        pd.DataFrame(
            {
                "Observations": [
                    ""  
                ]
            }
        ).to_excel(writer, sheet_name="Observations", index=False)


    print("\nProcessed datasets saved successfully at:")
    print(processed_path)

    # ---------- Model training ----------
    results = train_and_evaluate(
        get_models(),
        X_train, X_test,
        y_train, y_test
    )

    # ---------- Best Model selection ----------
    best_model = select_best_model(results)

    print("\n=========== OVERALL BEST MODEL (Healthcare Priority) ===========")
    print("Model Name :", best_model["Model"])
    print("Best Recall:", round(best_model["Recall"], 4))
    print("Best F1    :", round(best_model["F1"], 4))
    print("===============================================================")

    # ---------- Save model metrics ----------
    save_model_metrics(results, best_model)

    # ---------- Save artifacts ----------
    pickle.dump(
        best_model["Model_Object"],
        open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb")
    )

    pickle.dump(
        preprocessor,
        open(os.path.join(MODEL_DIR, "preprocessor.pkl"), "wb")
    )

    pickle.dump(
        scaler,
        open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb")
    )

    pickle.dump(
        list(X.columns),
        open(os.path.join(MODEL_DIR, "features.pkl"), "wb")
    )

    print("\nTRAINING COMPLETED")
    print("Best Model:", best_model["Model"])

if __name__ == "__main__":
    main()
