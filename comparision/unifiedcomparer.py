import pandas as pd
import matplotlib.pyplot as plt
import os

# ================================
# Paths to your metrics CSVs
# ================================
sentiment_csv = "sentiment/results_sentiment/model_metrics.csv"
hate_csv = "hate/results/model_metrics.csv"

# ================================
# Load CSVs
# ================================
sentiment_df = pd.read_csv(sentiment_csv)
hate_df = pd.read_csv(hate_csv)

# Rename columns to match
sentiment_df = sentiment_df.rename(columns={
    "ROC_AUC_Macro": "ROC_AUC"
})
hate_df = hate_df.rename(columns={
    "ROC_AUC": "ROC_AUC"
})

# Add a column to indicate the task
sentiment_df["Task"] = "Sentiment"
hate_df["Task"] = "Hate Speech"

# Add a prefix to model names to avoid confusion if same names exist
sentiment_df["Model"] = sentiment_df["Model"] + " (Sentiment)"
hate_df["Model"] = hate_df["Model"] + " (Hate)"

# ================================
# Combine both
# ================================
combined_df = pd.concat([sentiment_df, hate_df], ignore_index=True)

# Save combined CSV
os.makedirs("results_comparison", exist_ok=True)
combined_df.to_csv("results_comparison/combined_model_metrics.csv", index=False)

# ================================
# Plot bar chart
# ================================
metrics = ["Accuracy", "F1_Macro", "ROC_AUC"]

combined_df.set_index("Model")[metrics].plot(kind="bar", figsize=(12,6))
plt.title("Sentiment vs Hate Speech Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("results_comparison/combined_metric_comparison.png")
plt.close()

print("✅ Combined metrics CSV and bar chart saved in results_comparison/")
