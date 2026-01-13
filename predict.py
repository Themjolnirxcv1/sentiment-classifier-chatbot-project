import joblib
import numpy as np
import re




bundle = joblib.load("sentiment_bundle.pkl")
model = bundle["model"]
accuracy = bundle["accuracy"]
vectorizer = bundle["vectorizer"]
print("model accuracy:", bundle["accuracy"])
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text



while True:
    raw = input("masukan kalimat (atau 'ex'): ")
    if raw.lower() == "ex":
        print("udeh ye,'bye'")
        break
    text = clean_text(raw)    
    vec = vectorizer.transform([text])
    threshold = 0.6
    proba = model.predict_proba(vec)[0]
    idx = proba.argmax()
    label = model.classes_[idx]
    confidence = proba[idx]

    if confidence < threshold:
        print(f"prediksi: ambigu ({confidence:.2f})")

    else:
        print(f"prediksi : {label} ({confidence:.2f})")

        