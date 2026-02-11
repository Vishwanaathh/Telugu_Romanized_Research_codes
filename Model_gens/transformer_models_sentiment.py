import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

# -------------------------
# CONFIG
# -------------------------
CSV_PATH = "../Datasets/gpteacher_with_sentiment.csv"
TEXT_COLUMN = "telugu_transliterated_output"
LABEL_COLUMN = "vader_sentiment"

MODEL_SAVE_PATH = "tiny_transformer_sentiment"

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

print("Available columns:", df.columns)

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

# -------------------------
# ENCODE LABELS
# -------------------------
label_encoder = LabelEncoder()
df[LABEL_COLUMN] = label_encoder.fit_transform(df[LABEL_COLUMN])

# -------------------------
# CONVERT TO HF DATASET
# -------------------------
dataset = Dataset.from_pandas(df)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch[TEXT_COLUMN],
        padding="max_length",
        truncation=True,
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)

dataset = dataset.rename_column(LABEL_COLUMN, "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# -------------------------
# LOAD MODEL
# -------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

# -------------------------
# TRAINING ARGS
# -------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_strategy="no",
    logging_steps=100
)

# -------------------------
# TRAIN (NO TRAIN/TEST SPLIT)
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# -------------------------
# SAVE MODEL
# -------------------------
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("Tiny transformer trained and saved successfully.")
