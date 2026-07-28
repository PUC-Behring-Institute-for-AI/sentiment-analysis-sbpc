from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


DATASET = ROOT / "dataset" / "dataset.csv"

OUTPUT = ROOT / "train" / "saved_models"
OUTPUT.mkdir(exist_ok=True)

df = pd.read_csv(
    DATASET,
    usecols=["review_text", "polarity"]
)

df = df.dropna(subset=["review_text", "polarity"])
df["review_text"] = df["review_text"].astype(str).str.lower()

# Balanceia o dataset por undersampling

positive = df[df["polarity"] == 1.0]
negative = df[df["polarity"] == 0.0]

# Amostra aleatoriamente a mesma quantidade de positivos
positive = positive.sample(
    n=len(negative),
    random_state=42
)

# Junta os dois conjuntos
df = pd.concat([positive, negative])

# Embaralha as linhas
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print(f"Número de exemplos: {len(df)}")
print(df["polarity"].value_counts())


X = df["review_text"]
y = df["polarity"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline = Pipeline([
    (
        "vectorizer",
        CountVectorizer(),
    ),
    (
        "classifier",
        MultinomialNB(),
    ),
])

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred))

joblib.dump(
    pipeline,
    OUTPUT / "bow_pipeline.joblib"
)

print("Modelo salvo.")