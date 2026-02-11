import os
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================================================
# Create results folder
# =====================================================
os.makedirs("results", exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================
data_path = "../../Datasets/cleaned_training_data_telugu-hate.csv"
data = pd.read_csv(data_path)

le = LabelEncoder()
data["hate_encoded"] = le.fit_transform(data["Label"])

X_text = data["Comments"]
y_true = data["hate_encoded"]

print("Dataset loaded:", data.shape)

# =====================================================
# TF-IDF (must match training)
# =====================================================
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X_text)

# =====================================================
# Load Classical Models
# =====================================================
logreg = joblib.load("../../models/logregmodel.pkl")
rf = joblib.load("../../models/rfc.pkl")

# Train / Load SVM
# If you already trained and saved SVM, just load:
# svm = joblib.load("../../models/svm_model.pkl")
# Otherwise, train here:
svm = SVC(kernel="linear", probability=True, random_state=42)
svm.fit(X_tfidf, y_true)
joblib.dump(svm, "../../models/svm_model.pkl")
print("SVM trained and saved.")

# =====================================================
# Logistic Regression Predictions
# =====================================================
y_pred_lr = logreg.predict(X_tfidf)
y_prob_lr = logreg.predict_proba(X_tfidf)[:, 1]

# =====================================================
# Random Forest Predictions
# =====================================================
y_pred_rf = rf.predict(X_tfidf)
y_prob_rf = rf.predict_proba(X_tfidf)[:, 1]

# =====================================================
# SVM Predictions
# =====================================================
y_pred_svm = svm.predict(X_tfidf)
y_prob_svm = svm.predict_proba(X_tfidf)[:, 1]

# =====================================================
# Load Transformer Model
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("../../models/tiny_transformer_model")
model = AutoModelForSequenceClassification.from_pretrained(
    "../../models/tiny_transformer_model"
).to(device)
model.eval()


def get_transformer_predictions(texts, batch_size=16):
    preds = []
    probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size].tolist()
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

        preds.extend(torch.argmax(probabilities, dim=1).cpu().numpy())
        probs.extend(probabilities[:, 1].cpu().numpy())

    return np.array(preds), np.array(probs)


y_pred_tf, y_prob_tf = get_transformer_predictions(X_text)

# =====================================================
# Evaluation Function
# =====================================================
def evaluate_model(name, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Macro": f1_score(y_true, y_pred, average="macro"),
        "ROC_AUC": roc_auc_score(y_true, y_prob)
    }


# =====================================================
# Evaluate all models
# =====================================================
results = []
results.append(evaluate_model("Logistic Regression", y_true, y_pred_lr, y_prob_lr))
results.append(evaluate_model("Random Forest", y_true, y_pred_rf, y_prob_rf))
results.append(evaluate_model("SVM", y_true, y_pred_svm, y_prob_svm))
results.append(evaluate_model("Tiny Transformer", y_true, y_pred_tf, y_prob_tf))

results_df = pd.DataFrame(results)

print("\nModel Comparison:\n")
print(results_df)

# =====================================================
# Save Metrics CSV
# =====================================================
results_df.to_csv("results/model_metrics.csv", index=False)

# =====================================================
# Confusion Matrices
# =====================================================
def save_confusion_matrix(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"results/{name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()


for model_name, y_pred in zip(
    ["Logistic Regression", "Random Forest", "SVM", "Tiny Transformer"],
    [y_pred_lr, y_pred_rf, y_pred_svm, y_pred_tf]
):
    save_confusion_matrix(model_name, y_true, y_pred)

# =====================================================
# ROC Curve
# =====================================================
plt.figure()

fpr_lr, tpr_lr, _ = roc_curve(y_true, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_true, y_prob_rf)
fpr_svm, tpr_svm, _ = roc_curve(y_true, y_prob_svm)
fpr_tf, tpr_tf, _ = roc_curve(y_true, y_prob_tf)

plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot(fpr_svm, tpr_svm, label="SVM")
plt.plot(fpr_tf, tpr_tf, label="Tiny Transformer")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_comparison.png")
plt.close()

# =====================================================
# Bar Chart Comparison
# =====================================================
metrics_df = results_df.set_index("Model")[["Accuracy", "F1_Macro", "ROC_AUC"]]
metrics_df.plot(kind="bar")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("results/metric_comparison.png")
plt.close()

print("\n✅ All comparison results saved inside results/")
