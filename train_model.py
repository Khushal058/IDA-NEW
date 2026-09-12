"""
train_model.py
================
Builds the AI model for diagnosing Iron Deficiency Anaemia (IDA) using ONLY
patient demographics + signs & symptoms (NOT lab test values), following
the methodology described in the research proposal step by step:

  1. Data Preprocessing      - missing values, encoding check, scaling
  2. Feature Engineering     - select relevant signs/symptoms only
  3. Model Selection         - compare Decision Tree, Random Forest, SVM,
                                and Neural Network candidates
  4. Model Training          - train / validation / test split,
                                hyperparameter tuning
  5. Evaluation & Validation - accuracy, precision, recall, F1, AUC-ROC,
                                5-fold cross-validation for robustness
  6. Explainability           - feature importance + SHAP values

Run this file ONCE first. It creates:
    - ida_model.joblib            (the final, best-performing trained model)
    - feature_list.joblib         (the exact list/order of inputs the model expects)
    - scaler.joblib                (feature scaler, used only if the winning
                                    model needs scaled inputs, e.g. SVM/Neural Net)
    - model_meta.joblib           (records which model won and whether it
                                    needs scaled inputs, so the GUI apps know
                                    how to use it correctly)
    - feature_importance.png      (chart: which symptoms matter most)
    - shap_summary.png            (chart: SHAP explainability - how each
                                    symptom pushes the prediction up/down)
    - model_comparison.png        (chart: how the 4 candidate models compared)
    - model_report.txt            (full performance + methodology report)

After this finishes successfully, run gui_app.py or web_app.py to use the model.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")  # no display needed to save charts
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
)

REPORT_LINES = []
def log(msg):
    """Print to screen AND remember for the final report file."""
    print(msg)
    REPORT_LINES.append(msg)


# ===========================================================================
# STEP 1: Load the data
# ===========================================================================
log("=" * 70)
log("STEP 1: Loading the combined dataset")
log("=" * 70)
df = pd.read_csv("combined_dataset.csv")
log(f"Loaded {len(df)} patient records, {df.shape[1]} columns.")


# ===========================================================================
# STEP 2: Data Preprocessing
# (per proposal: "Clean and preprocess... handle missing values, outliers,
#  and inconsistencies. Convert categorical variables into numerical
#  representations. Normalize or standardize numerical features.")
# ===========================================================================
log("\n" + "=" * 70)
log("STEP 2: Data Preprocessing")
log("=" * 70)

missing_count = df.isnull().sum().sum()
log(f"Missing values found across dataset: {missing_count}")
if missing_count > 0:
    df = df.fillna(df.median(numeric_only=True))
    log("-> Missing numeric values were filled with the column median.")
else:
    log("-> No missing values found; no imputation needed.")

duplicate_count = df.duplicated().sum()
log(f"Duplicate records found: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()
    log(f"-> Duplicates removed. Remaining records: {len(df)}")

log("All categorical variables (Gender, Address, Education, Occupation, "
    "Income, and every symptom) are already numerically coded per the "
    "study's data-coding sheet (e.g. Gender: 1=Male, 2=Female), so no "
    "additional text-to-number conversion is required.")


# ===========================================================================
# STEP 3: Feature Engineering / Selection
# (per proposal AIM: diagnosis based on "signs and symptoms only" -- so lab
#  values are intentionally excluded from the model's inputs, even though
#  they exist in the dataset as the original clinical confirmation.)
# ===========================================================================
log("\n" + "=" * 70)
log("STEP 3: Feature Engineering / Selection")
log("=" * 70)

EXCLUDED_COLUMNS = ["RecordID", "Outcome"] + [c for c in df.columns if c.startswith(("cbc_", "ps_"))]
FEATURE_COLUMNS = [c for c in df.columns if c not in EXCLUDED_COLUMNS]

log(f"Using {len(FEATURE_COLUMNS)} input features (demographics + signs/symptoms only):")
log("   " + ", ".join(FEATURE_COLUMNS))
log(f"Lab test columns ({len(EXCLUDED_COLUMNS) - 2} CBC/Peripheral-Smear "
    "parameters) are EXCLUDED from inputs, per the study's aim of diagnosis without "
    "requiring a blood test. Outcome (case/control) was generated using the exact same "
    "classify_lab_verdict() core-panel rule the apps use to confirm IDA from a real "
    "patient's lab report, so training stays in sync with the app's lab-confirmation logic.")

X = df[FEATURE_COLUMNS]
y = df["Outcome"]


# ===========================================================================
# STEP 4: Train / Validation / Test split
# (per proposal: "Split the dataset into training, validation, and test
#  sets to evaluate the model's performance.")
# ===========================================================================
log("\n" + "=" * 70)
log("STEP 4: Splitting into Train / Validation / Test sets")
log("=" * 70)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
log(f"Training set  : {len(X_train)} records (70%) - used to train each model")
log(f"Validation set: {len(X_val)} records (15%) - used to compare/tune models")
log(f"Test set      : {len(X_test)} records (15%) - held back, used ONLY for the final report")

# Scaled versions (needed for SVM and Neural Network; tree models don't need this)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
log("Numeric features were standardized (mean=0, std=1) for the SVM and "
    "Neural Network candidates, which are sensitive to feature scale. "
    "Tree-based models (Decision Tree, Random Forest) use the raw values, "
    "since they don't require scaling.")


# ===========================================================================
# STEP 5: Model Selection - train and compare multiple candidate algorithms
# (per proposal: "Decision trees, random forests, support vector machines,
#  and neural networks are potential candidates")
# ===========================================================================
log("\n" + "=" * 70)
log("STEP 5: Model Selection - comparing candidate algorithms")
log("=" * 70)

candidates = {
    "Decision Tree": (DecisionTreeClassifier(max_depth=10, random_state=42), X_train, X_val),
    "Random Forest": (RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42,
                                              class_weight="balanced", n_jobs=-1), X_train, X_val),
    "Support Vector Machine": (SVC(kernel="rbf", probability=True, random_state=42), X_train_scaled, X_val_scaled),
    "Neural Network (MLP)": (MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400, random_state=42), X_train_scaled, X_val_scaled),
}

comparison_results = {}
for name, (clf, X_tr, X_va) in candidates.items():
    clf.fit(X_tr, y_train)
    val_pred = clf.predict(X_va)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred)
    comparison_results[name] = {"model": clf, "val_accuracy": val_acc, "val_f1": val_f1}
    log(f"  {name:<26} -> Validation Accuracy: {val_acc:.4f} | Validation F1: {val_f1:.4f}")

# Pick the best model by validation F1-score
best_name = max(comparison_results, key=lambda n: comparison_results[n]["val_f1"])
log(f"\nBest performing model on the validation set: {best_name}")

# Save a comparison bar chart
plt.figure(figsize=(8, 5))
names = list(comparison_results.keys())
accs = [comparison_results[n]["val_accuracy"] for n in names]
f1s = [comparison_results[n]["val_f1"] for n in names]
x_pos = np.arange(len(names))
bar_width = 0.32
plt.bar(x_pos - bar_width/2 - 0.03, accs, width=bar_width, label="Validation Accuracy", color="#3b6e8f")
plt.bar(x_pos + bar_width/2 + 0.03, f1s, width=bar_width, label="Validation F1-score", color="#8fb9cf")
plt.xticks(x_pos, names, rotation=15, ha="right")
plt.ylim(0, 1.18)
plt.title("Model Comparison on Validation Set")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=130)
log("Saved model_comparison.png")


# ===========================================================================
# STEP 6: Hyperparameter Tuning (for the winning model)
# (per proposal: "Tune hyperparameters to optimize the model's performance.")
# ===========================================================================
log("\n" + "=" * 70)
log(f"STEP 6: Hyperparameter Tuning for {best_name}")
log("=" * 70)

uses_scaled = best_name in ("Support Vector Machine", "Neural Network (MLP)")
X_train_full = X_train_scaled if uses_scaled else X_train

if best_name == "Random Forest":
    param_grid = {"n_estimators": [200, 300], "max_depth": [10, 12, 16]}
    base_model = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
elif best_name == "Decision Tree":
    param_grid = {"max_depth": [6, 10, 14], "min_samples_split": [2, 5, 10]}
    base_model = DecisionTreeClassifier(random_state=42)
elif best_name == "Support Vector Machine":
    param_grid = {"C": [1, 10], "gamma": ["scale", "auto"]}
    base_model = SVC(kernel="rbf", probability=True, random_state=42)
else:  # Neural Network
    param_grid = {"hidden_layer_sizes": [(32, 16), (64, 32)], "alpha": [0.0001, 0.001]}
    base_model = MLPClassifier(max_iter=400, random_state=42)

log(f"Searching over: {param_grid}")
grid = GridSearchCV(base_model, param_grid, scoring="f1", cv=3, n_jobs=-1)
grid.fit(X_train_full, y_train)
log(f"Best hyperparameters found: {grid.best_params_}")
log(f"Best cross-validated F1 during search: {grid.best_score_:.4f}")

final_model = grid.best_estimator_
final_uses_scaled = uses_scaled


# ===========================================================================
# STEP 7: Cross-Validation (robustness check)
# (per proposal: "Implement cross-validation techniques to ensure the
#  model's robustness.")
# ===========================================================================
log("\n" + "=" * 70)
log("STEP 7: 5-Fold Cross-Validation (robustness check)")
log("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
if final_uses_scaled:
    X_train_val = pd.DataFrame(
        np.vstack([X_train_scaled, X_val_scaled]), columns=FEATURE_COLUMNS
    )
else:
    X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val])
cv_scores = cross_val_score(final_model, X_train_val, y_train_val, cv=cv, scoring="accuracy", n_jobs=-1)
log(f"5-fold CV accuracy scores: {[round(s, 4) for s in cv_scores]}")
log(f"Mean CV accuracy: {cv_scores.mean():.4f}  (std: {cv_scores.std():.4f})")
log("A low standard deviation across folds indicates the model performs "
    "consistently and is not just well-tuned to one particular data split.")


# ===========================================================================
# STEP 8: Final Training + Evaluation on the held-out Test set
# (per proposal: "Assess the model's performance using appropriate
#  evaluation metrics such as accuracy, precision, recall, F1-score, and
#  AUC-ROC.")
# ===========================================================================
log("\n" + "=" * 70)
log("STEP 8: Final Evaluation on the held-out Test set")
log("=" * 70)

# Retrain the tuned model on train+validation combined, test on the untouched test set
final_model.fit(X_train_val, y_train_val)
X_test_final = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS) if final_uses_scaled else X_test

y_pred = final_model.predict(X_test_final)
y_proba = final_model.predict_proba(X_test_final)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["No IDA", "IDA"])

# ---------------------------------------------------------------------------
# Clinical evaluation metrics — specifically requested by Dr. Ruchita Dixit's
# research protocol (Sensitivity, Specificity, PPV, NPV, ROC curve/AUC).
# These are standard diagnostic-test metrics used in clinical research,
# computed here from the confusion matrix:
#   cm = [[TN, FP],
#         [FN, TP]]
# ---------------------------------------------------------------------------
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")   # = Recall
specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")           # = Precision
npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

log(f"Final model      : {best_name}")
log(f"Accuracy         : {acc:.4f}")
log(f"Precision (PPV)  : {prec:.4f}")
log(f"Recall (Sens.)   : {rec:.4f}")
log(f"F1-score         : {f1:.4f}")
log(f"AUC-ROC          : {auc:.4f}")
log(f"Sensitivity      : {sensitivity:.4f}   (True Positive Rate — % of actual IDA cases correctly identified)")
log(f"Specificity      : {specificity:.4f}   (True Negative Rate — % of actual non-IDA cases correctly identified)")
log(f"PPV              : {ppv:.4f}   (Of those predicted IDA, % that truly are IDA)")
log(f"NPV              : {npv:.4f}   (Of those predicted non-IDA, % that truly are non-IDA)")
log(f"Confusion matrix:\n{cm}")
log("\n" + report)

# Save a clinical metrics summary file separately too, for easy reference
with open("clinical_metrics.txt", "w") as f:
    f.write("Clinical Evaluation Metrics — Iron Deficiency Anaemia AI Model\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Model used: {best_name}\n\n")
    f.write(f"Accuracy     : {acc:.4f}\n")
    f.write(f"Sensitivity  : {sensitivity:.4f}  (Recall / True Positive Rate)\n")
    f.write(f"Specificity  : {specificity:.4f}  (True Negative Rate)\n")
    f.write(f"PPV          : {ppv:.4f}  (Precision / Positive Predictive Value)\n")
    f.write(f"NPV          : {npv:.4f}  (Negative Predictive Value)\n")
    f.write(f"F1-score     : {f1:.4f}\n")
    f.write(f"AUC-ROC      : {auc:.4f}\n\n")
    f.write("Confusion Matrix (rows=actual, cols=predicted)\n")
    f.write("                 Pred: No IDA   Pred: IDA\n")
    f.write(f"Actual: No IDA   {tn:<14}{fp}\n")
    f.write(f"Actual: IDA      {fn:<14}{tp}\n\n")
    f.write("Definitions:\n")
    f.write("- Sensitivity (Recall): Of all patients who truly have IDA, what % did the model correctly flag?\n")
    f.write("- Specificity: Of all patients who truly do NOT have IDA, what % did the model correctly clear?\n")
    f.write("- PPV (Precision): Of all patients the model flagged as IDA, what % actually have it?\n")
    f.write("- NPV: Of all patients the model cleared as non-IDA, what % actually don't have it?\n")
log("Saved clinical_metrics.txt")

# ---------------------------------------------------------------------------
# ROC Curve plot (requested in the research protocol)
# ---------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6.5, 6))
plt.plot(fpr, tpr, color="#1D9E75", linewidth=2.2, label=f"ROC curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1, label="Random guess (AUC = 0.5)")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve — IDA Prediction Model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=130)
log("Saved roc_curve.png")


# ===========================================================================
# STEP 9: Explainability
# (per proposal: "Techniques like feature importance, SHAP... or LIME can
#  help provide insights into why the model makes certain predictions.")
# ===========================================================================
log("\n" + "=" * 70)
log("STEP 9: Explainability (Feature Importance + SHAP)")
log("=" * 70)

if hasattr(final_model, "feature_importances_"):
    importances = pd.Series(final_model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    plt.figure(figsize=(9, 10))
    importances.plot(kind="barh", color="#3b6e8f")
    plt.title("Which symptoms matter most for predicting IDA?")
    plt.xlabel("Importance score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=130)
    log("Saved feature_importance.png")
else:
    log("Selected model does not expose feature_importances_ directly; "
        "skipping the basic importance chart (SHAP chart below covers this).")

try:
    log("Computing SHAP values (this explains each prediction symptom-by-symptom)...")
    sample = X_test.sample(n=min(300, len(X_test)), random_state=42)
    sample_for_model = scaler.transform(sample) if final_uses_scaled else sample

    if best_name in ("Random Forest", "Decision Tree"):
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(sample_for_model)
        if isinstance(shap_values, list):
            sv = shap_values[1]               # older shap: list per class
        elif shap_values.ndim == 3:
            sv = shap_values[:, :, 1]         # newer shap: (samples, features, classes)
        else:
            sv = shap_values
    else:
        background = X_train_full[:100]
        explainer = shap.KernelExplainer(final_model.predict_proba, background)
        shap_values = explainer.shap_values(sample_for_model[:100], nsamples=100)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values
        sample = sample.iloc[:100]

    plt.figure()
    shap.summary_plot(sv, sample, feature_names=FEATURE_COLUMNS, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()
    log("Saved shap_summary.png")
except Exception as e:
    log(f"SHAP chart could not be generated ({e}); feature_importance.png still covers explainability.")


# ===========================================================================
# STEP 10: Save the final model
# ===========================================================================
log("\n" + "=" * 70)
log("STEP 10: Saving the final model to disk")
log("=" * 70)

joblib.dump(final_model, "ida_model.joblib")
joblib.dump(FEATURE_COLUMNS, "feature_list.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump({"name": best_name, "uses_scaled_input": final_uses_scaled}, "model_meta.joblib")
log("Saved ida_model.joblib, feature_list.joblib, scaler.joblib, model_meta.joblib")

with open("model_report.txt", "w") as f:
    f.write("\n".join(REPORT_LINES))
log("\nFull step-by-step report saved to model_report.txt")
log("\nAll done! You can now run: python gui_app.py   OR   streamlit run web_app.py")
