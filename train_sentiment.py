import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import re

#Data Cleaning

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text
# load data
df = pd.read_csv("data/sentiment.csv", sep=";")
df.columns = df.columns.str.strip().str.replace(",", "")
df["text"] = df["text"].apply(clean_text)

print(df.columns)
print(df.head(2))
print(df["label"].value_counts())

# split
x_train, x_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))
x_train_vec = vectorizer.fit_transform(x_train)

# model
model = LogisticRegression(
    class_weight="balanced",
    C=5.0,
    solver="liblinear"
)
model.fit(x_train_vec, y_train)

# evaluation
x_test_vec = vectorizer.transform(x_test)
y_pred = model.predict(x_test_vec)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# save model
joblib.dump({
    "model": model,
    "vectorizer": vectorizer,
    "accuracy": model.score(x_test_vec, y_test),
    "classes": model.classes_.tolist()
}, "sentiment_bundle.pkl")

print("Model & vectorizer disimpan.")

# interpretability
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]

top_pos = sorted(zip(feature_names, coef), key=lambda x: x[1], reverse=True)[:10]
top_neg = sorted(zip(feature_names, coef), key=lambda x: x[1])[:10]

print("\nTOP POSITIVE FEATURES:")
for w, c in top_pos:
    print(w, round(c, 3))

print("\nTOP NEGATIVE FEATURES:")
for w, c in top_neg:
    print(w, round(c, 3))

