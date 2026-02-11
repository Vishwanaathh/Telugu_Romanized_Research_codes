import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("../Datasets/gpteacher_trimmedd.csv")

TEXT_COLUMN = "telugu_transliterated_output"

analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    if not isinstance(text, str):
        return "neutral"
    
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

df["vader_sentiment"] = df[TEXT_COLUMN].apply(get_vader_sentiment)

df.to_csv("../Datasets/gpteacher_with_sentiment.csv", index=False)

print("✅ Sentiment column added successfully.")
print(df["vader_sentiment"].value_counts())
