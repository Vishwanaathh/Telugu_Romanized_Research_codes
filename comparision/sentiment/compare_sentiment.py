import os
import time
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
    confusion_matrix,
    ConfusionMatrixDisplay
)

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# =========================================================
# PATHS
# =========================================================
DATA_PATH = "../../Datasets/gpteacher_with_sentiment.csv"
MODEL_PATH = "../../models"
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "tiny_transformer_sentiment")
RESULTS_PATH = "results_sentiment"

os.makedirs(RESULTS_PATH, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

TEXT_COLUMN = "telugu_transliterated_output"
LABEL_COLUMN = "vader_sentiment"

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
X_text = df[TEXT_COLUMN].astype(str)
y_labels = df[LABEL_COLUMN]

print(f"Dataset loaded: {len(df)} samples")

# =========================================================
# ENCODE LABELS
# =========================================================
print("Encoding labels...")
label_encoder = LabelEncoder()
y_true = label_encoder.fit_transform(y_labels)
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

# =========================================================
# LOAD TF-IDF
# =========================================================
print("Loading TF-IDF vectorizer...")
tfidf = joblib.load(os.path.join(MODEL_PATH, "sentiment_tfidf.pkl"))
X_tfidf = tfidf.transform(X_text)
print("TF-IDF transformation complete")

# =========================================================
# LOAD CLASSICAL MODELS
# =========================================================
print("Loading Logistic Regression...")
logreg = joblib.load(os.path.join(MODEL_PATH, "sentiment_logreg.pkl"))

print("Loading Random Forest...")
rf = joblib.load(os.path.join(MODEL_PATH, "sentiment_rf.pkl"))

# =========================================================
# TRAIN/LOAD SVM
# =========================================================
svm_path = os.path.join(MODEL_PATH, "sentiment_svm.pkl")
if os.path.exists(svm_path):
    print("Loading pre-trained SVM...")
    svm = joblib.load(svm_path)
else:
    print("Training new SVM...")
    svm = SVC(kernel="linear", probability=True, random_state=42)
    svm.fit(X_tfidf, y_true)
    joblib.dump(svm, svm_path)
    print("SVM trained and saved.")

# =========================================================
# CLASSICAL MODEL PREDICTIONS
# =========================================================
print("Running Logistic Regression...")
y_pred_lr = logreg.predict(X_tfidf)
y_prob_lr = logreg.predict_proba(X_tfidf)

print("Running Random Forest...")
y_pred_rf = rf.predict(X_tfidf)
y_prob_rf = rf.predict_proba(X_tfidf)

print("Running SVM...")
y_pred_svm = svm.predict(X_tfidf)
y_prob_svm = svm.predict_proba(X_tfidf)

# =========================================================
# LOAD TRANSFORMER
# =========================================================
print("Loading Tiny Transformer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained(TRANSFORMER_PATH)
transformer = DistilBertForSequenceClassification.from_pretrained(
    TRANSFORMER_PATH
).to(device)
transformer.eval()

def get_transformer_predictions(texts, batch_size=32):
    preds = []
    probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size].tolist()
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = transformer(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

        preds.extend(torch.argmax(probabilities, dim=1).cpu().numpy())
        probs.extend(probabilities.cpu().numpy())

    return np.array(preds), np.array(probs)

print("Running Transformer inference...")
y_pred_tf, y_prob_tf = get_transformer_predictions(X_text)
print("Transformer inference complete")

# =========================================================
# METRICS FUNCTION
# =========================================================
def evaluate(name, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_macro": precision_score(y_true, y_pred, average="macro"),
        "Recall_macro": recall_score(y_true, y_pred, average="macro"),
        "F1_macro": f1_score(y_true, y_pred, average="macro"),
        "ROC_AUC_macro": roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    }

# =========================================================
# EVALUATION
# =========================================================
results = []
results.append(evaluate("Logistic Regression", y_true, y_pred_lr, y_prob_lr))
results.append(evaluate("Random Forest", y_true, y_pred_rf, y_prob_rf))
results.append(evaluate("SVM", y_true, y_pred_svm, y_prob_svm))
results.append(evaluate("Tiny Transformer", y_true, y_pred_tf, y_prob_tf))

results_df = pd.DataFrame(results)
print("\n===== MODEL COMPARISON =====\n")
print(results_df)

results_df.to_csv(os.path.join(RESULTS_PATH, "model_metrics.csv"), index=False)

# =========================================================
# CONFUSION MATRICES
# =========================================================
def save_confusion(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f"{name.replace(' ', '_')}_cm.png"))
    plt.close()

for model_name, y_pred in zip(
    ["Logistic Regression", "Random Forest", "SVM", "Tiny Transformer"],
    [y_pred_lr, y_pred_rf, y_pred_svm, y_pred_tf]
):
    save_confusion(model_name, y_true, y_pred)

# =========================================================
# BAR GRAPH COMPARISON
# =========================================================
metrics_df = results_df.set_index("Model")[["Accuracy", "F1_macro", "ROC_AUC_macro"]]
metrics_df.plot(kind="bar")
plt.title("Sentiment Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "metric_comparison.png"))
plt.close()

print("\nAll results saved inside results_sentiment/")
print("DONE.")
