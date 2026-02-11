import os
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# --------------------------------
# Create models directory
# --------------------------------
os.makedirs("../models", exist_ok=True)

# --------------------------------
# Load Dataset
# --------------------------------
df = pd.read_csv("../Datasets/gpteacher_with_sentiment.csv")

TEXT_COLUMN = "telugu_transliterated_output"
LABEL_COLUMN = "vader_sentiment"

df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])

# --------------------------------
# Encode Labels
# --------------------------------
le = LabelEncoder()
y = le.fit_transform(df[LABEL_COLUMN])

# --------------------------------
# TF-IDF Vectorization
# --------------------------------
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df[TEXT_COLUMN])

# --------------------------------
# Logistic Regression
# --------------------------------
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X, y)

joblib.dump(logreg, "../models/sentiment_logreg.pkl")

# --------------------------------
# Random Forest
# --------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

joblib.dump(rf, "../models/sentiment_rf.pkl")

# --------------------------------
# Save Encoder + Vectorizer
# --------------------------------
joblib.dump(tfidf, "../models/sentiment_tfidf.pkl")
joblib.dump(le, "../models/sentiment_label_encoder.pkl")

print("✅ Models trained on FULL dataset and saved successfully.")
