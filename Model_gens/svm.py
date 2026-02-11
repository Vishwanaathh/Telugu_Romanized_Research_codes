# svm_training.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
import os

# ==============================
# Configuration
# ==============================
MAX_FEATURES = 5000
RANDOM_STATE = 42
MODELS_DIR = "../models/svm_models"  # separate folder for SVM
os.makedirs(MODELS_DIR, exist_ok=True)

# ==============================
# Function to train SVM
# ==============================
def train_svm(df, text_col, label_col, task_name="task"):
    print(f"\n=== Training SVM for {task_name} ===")

    # Encode labels
    le = LabelEncoder()
    df[label_col + "_encoded"] = le.fit_transform(df[label_col])
    
    X_text = df[text_col]
    y = df[label_col + "_encoded"]

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
    X = tfidf.fit_transform(X_text)

    # Save TF-IDF
    tfidf_path = os.path.join(MODELS_DIR, f"{task_name}_tfidf.pkl")
    joblib.dump(tfidf, tfidf_path)

    # Train SVM
    svm = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    svm.fit(X, y)
    
    # Save SVM model
    svm_path = os.path.join(MODELS_DIR, f"{task_name}_svm.pkl")
    joblib.dump(svm, svm_path)

    print(f"SVM model and TF-IDF saved for {task_name}:")
    print(f"  - TF-IDF: {tfidf_path}")
    print(f"  - SVM: {svm_path}")

    return le, tfidf, svm

# ==============================
# Hate Speech Dataset
# ==============================
hate_df = pd.read_csv('../Datasets/cleaned_training_data_telugu-hate.csv')
le_hate, tfidf_hate, svm_hate = train_svm(
    hate_df,
    text_col="Comments",
    label_col="Label",
    task_name="hate"
)


sentiment_df = pd.read_csv('../Datasets/gpteacher_with_sentiment.csv')
sentiment_df.columns = sentiment_df.columns.str.strip()
sentiment_df = sentiment_df[['telugu_transliterated_output', 'vader_sentiment']].dropna()

le_sent, tfidf_sent, svm_sent = train_svm(
    sentiment_df,
    text_col="telugu_transliterated_output",
    label_col="vader_sentiment",
    task_name="sentiment"
)

print("\n✅ All SVM models trained and saved successfully.")
