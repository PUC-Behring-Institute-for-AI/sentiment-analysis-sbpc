from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import set_seed


# ==========================================================
# Configurações
# ==========================================================

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

MAX_LENGTH = 128

BATCH_SIZE = 16

RANDOM_STATE = 42
set_seed(RANDOM_STATE)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Usando dispositivo: {DEVICE}")

DATASET = ROOT / "dataset" / "dataset.csv"

OUTPUT = ROOT / "train" / "saved_models" / "bertimbau"

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# Leitura do dataset
# ==========================================================

df = pd.read_csv(
    DATASET,
    usecols=[
        "review_text",
        "polarity"
    ]
)

df = df.dropna(
    subset=[
        "review_text",
        "polarity"
    ]
)

df["review_text"] = (
    df["review_text"]
    .astype(str)
    .str.lower()
)

print("\nDataset original")
print(df["polarity"].value_counts())


# ==========================================================
# Balanceamento (igual ao BoW)
# ==========================================================
df["polarity"] = df["polarity"].astype(int)

positive = df[
    df["polarity"] == 1
]

negative = df[
    df["polarity"] == 0
]

positive = positive.sample(
    n=len(negative),
    random_state=RANDOM_STATE
)

df = pd.concat(
    [
        positive,
        negative
    ]
)

df = (
    df
    .sample(
        frac=1,
        random_state=RANDOM_STATE
    )
    .reset_index(drop=True)
)

print("\nDataset balanceado")
print(df["polarity"].value_counts())

print(f"\nTotal de exemplos: {len(df):,}")


# ==========================================================
# Train / Test
# ==========================================================

train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["polarity"])


print(f"\nTreino: {len(train_df):,}")
print(f"Teste : {len(test_df):,}")


# ==========================================================
# Tokenizer
# ==========================================================

print("\nCarregando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer carregado.")

# Tokenize function
def tokenize_data(examples):
    return tokenizer(examples['review_text'], padding='max_length', truncation=True, max_length=MAX_LENGTH)

# Convertendo DataFrames para Datasets do Hugging Face
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenizar datasets
train_dataset = train_dataset.map(tokenize_data, batched=True, batch_size=BATCH_SIZE)
test_dataset = test_dataset.map(tokenize_data, batched=True, batch_size=BATCH_SIZE)

# Remover colunas desnecessárias e renomear a coluna de rótulos
train_dataset = train_dataset.remove_columns(["review_text"]).rename_column("polarity", "labels")
test_dataset = test_dataset.remove_columns(["review_text"]).rename_column("polarity", "labels")
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# ==========================================================
# Modelo
# ==========================================================

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

print("Modelo carregado.")

# ==========================================================
# Métricas
# ==========================================================

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ==========================================================
# Argumentos de treinamento
# ==========================================================

training_args = TrainingArguments(
    output_dir=OUTPUT,
    evaluation_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,
    weight_decay=0.01,
    learning_rate=2e-5,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_strategy="epoch",
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
    report_to='none'
)


# ==========================================================
# Trainer
# ==========================================================

trainer = Trainer(
    tokenizer=tokenizer,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# ==========================================================
# Treinar & salvar
# ==========================================================

trainer.train()
trainer.save_model(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
