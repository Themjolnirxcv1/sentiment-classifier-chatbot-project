import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


# Load Data
df = pd.read_csv("data/intent.csv", sep=";")
df.columns = ["text", "intent"]
print(df.isna().sum())

df["text"] = df["text"].fillna("")
df["text"] = df["text"].astype(str)
df["text"] = df["text"].apply(clean_text)

# Hapus baris dengan missing values di intent
df = df.dropna(subset=["intent"])

#split
x_train, x_test, y_train, y_test = train_test_split(
    df["text"],
    df["intent"],
    test_size=0.2,
    random_state=42,
    stratify=df["intent"]
)

#vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))
x_train_vec = vectorizer.fit_transform(x_train)

#model
model = LogisticRegression(
    class_weight="balanced",
    C=5.0,
    solver="lbfgs",
    max_iter=1000
)
model.fit(x_train_vec, y_train)

#evaluation
x_test_vec = vectorizer.transform(x_test)
y_pred = model.predict(x_test_vec)

print(classification_report(y_test, y_pred))    

#save model
joblib.dump(model, "intent_model.pkl")
joblib.dump(vectorizer, "intent_vectorizer.pkl")

print("Model & vectorizer disimpan.")



