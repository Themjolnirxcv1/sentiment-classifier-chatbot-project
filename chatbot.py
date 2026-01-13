import joblib
import random
import re

# ========= LOAD MODEL =========
intent_model = joblib.load("intent_model.pkl")
intent_vectorizer = joblib.load("intent_vectorizer.pkl")

sentiment_bundle = joblib.load("sentiment_bundle.pkl")
sentiment_model = sentiment_bundle["model"]
sentiment_vectorizer = sentiment_bundle["vectorizer"]



# ========= CLEAN TEXT =========
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# ========= RESPONSE BANK =========
responses = {
    "greeting": {
        "positive": ["Halo 👋", "Yo, kelihatannya lu lagi oke."],
        "negative": ["Halo. Lu keliatan capek.", "Hai. Ada apa?"],
        "neutral": ["Halo.", "Yo."]
    },
    "curhat": {
        "negative": [
            "Gue dengerin. Lu ga lebay.",
            "Capek itu valid. Lanjut ceritain."
        ],
        "positive": [
            "Mantap. Cerita yang bagus tuh.",
        ],
        "neutral": [
            "Gue di sini. Ceritain aja."
        ]
    },
    "question": {
        "neutral": [
            "Oke, tanya aja pelan-pelan.",
            "Gue coba jawab sebisanya."
        ]
    },
    "thanks": {
        "positive": ["Santai 👍", "Sama-sama."],
        "neutral": ["aman aja.", "Yoi."]

    }
}

confidence_threshold = 0.6

print("Chatbot hidup. Ketik 'ex' buat cabut.")

# ========= CONTEXT =========
context = {
    "last_intent": None
}

while True:
    user_input = input("lau: ")
    if user_input.lower() == "ex":
        print("bot: cabut dulu. jaga kepala lu.")
        break

    clean_input = clean_text(user_input)

    # ===== INTENT =====
    vec_intent = intent_vectorizer.transform([clean_input])
    intent_proba = intent_model.predict_proba(vec_intent)[0]
    intent_idx = intent_proba.argmax()
    intent = intent_model.classes_[intent_idx]
    intent_conf = intent_proba[intent_idx]

    if intent_conf < confidence_threshold:
        print("bot: Gue nangkepnya setengah-setengah. Bisa jelasin dikit?")
        continue

    # ===== SENTIMENT =====
    vec_sent = sentiment_vectorizer.transform([clean_input])
    sent_proba = sentiment_model.predict_proba(vec_sent)[0]
    sent_idx = sent_proba.argmax()
    sentiment = sentiment_model.classes_[sent_idx]


   # ===== RESPONSE BANK =====
    intent_responses = responses.get(intent, {})
    sentiment_responses = intent_responses.get(
        sentiment,
        intent_responses.get("neutral", ["Gue dengerin."])
    )

    # ===== CONTEXT OVERRIDE =====
    if intent == "curhat" and context["last_intent"] == "curhat":
        response = random.choice([
            "Gue masih di sini. Lanjut.",
            "Pelan-pelan. Ga perlu kuat sendirian."
        ])
    else:
        response = random.choice(sentiment_responses)

    print(f"bot: {response}")
    context["last_intent"] = intent

