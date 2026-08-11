"""
Trains the intent classifier from data/intents.json and saves the
resulting model artifacts to models/.

Run this once before starting the API or UI:
    python train.py
"""

import json
import os

from sklearn.model_selection import train_test_split

from src.config import (
    CENTROIDS_PATH,
    CLASSIFIER_PATH,
    FALLBACK_TAG,
    INTENTS_PATH,
    LABEL_ENCODER_PATH,
    MODELS_DIR,
)
from src.intent_classifier import IntentClassifier
from src.preprocessing import clean_text


def load_training_data() -> tuple[list[str], list[str]]:
    """Flattens intents.json into (text, label) training pairs."""
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        tag = intent["tag"]
        if tag == FALLBACK_TAG:
            continue  # fallback has no patterns; it's triggered by low confidence
        for pattern in intent["patterns"]:
            texts.append(clean_text(pattern))
            labels.append(tag)

    return texts, labels


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading training data from intents.json ...")
    texts, labels = load_training_data()
    print(f"Loaded {len(texts)} training examples across {len(set(labels))} intents.")

    # Hold out a validation split purely to sanity-check during training.
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    print("Encoding text and training classifier (this may take a minute)...")
    classifier = IntentClassifier()
    classifier.fit(X_train, y_train)

    # Quick validation check
    correct = 0
    for text, true_label in zip(X_val, y_val):
        predicted_label, confidence = classifier.predict(text)
        if predicted_label == true_label:
            correct += 1
    val_accuracy = correct / len(X_val) if X_val else 0.0
    print(f"Validation accuracy on held-out training split: {val_accuracy:.2%}")

    classifier.save(CLASSIFIER_PATH, LABEL_ENCODER_PATH, CENTROIDS_PATH)
    print(f"Model saved to: {CLASSIFIER_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")
    print(f"Intent centroids saved to: {CENTROIDS_PATH}")
    print("\nTraining complete. Run 'python -m src.evaluate' for a full evaluation report.")


if __name__ == "__main__":
    main()
