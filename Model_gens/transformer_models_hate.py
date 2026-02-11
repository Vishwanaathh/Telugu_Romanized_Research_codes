import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder

hate = pd.read_csv('../Datasets/cleaned_training_data_telugu-hate.csv')

le = LabelEncoder()
hate['hate_encoded'] = le.fit_transform(hate['Label'])

X = hate['Comments']
Y = hate['hate_encoded']

class TeluguDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)

train_dataset = TeluguDataset(X, Y, tokenizer)

training_args = TrainingArguments(
    output_dir='./tiny_transformer_results',
    num_train_epochs=2,
    per_device_train_batch_size=4,
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

model.save_pretrained("./tiny_transformer_model")
tokenizer.save_pretrained("./tiny_transformer_model")

print("✅ Tiny transformer training complete and saved to ./tiny_transformer_model")
