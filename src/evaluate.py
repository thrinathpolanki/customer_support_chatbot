"""
Evaluates the trained intent classifier on a held-out test set
(data/test_data.json) and prints/saves an accuracy + classification
report, exactly what you'd show a mentor to prove the model works.
"""

import json

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import (
    CENTROIDS_PATH,
    CLASSIFIER_PATH,
    EVAL_REPORT_PATH,
    LABEL_ENCODER_PATH,
    TEST_DATA_PATH,
)
from src.intent_classifier import IntentClassifier
from src.preprocessing import clean_text


def run_evaluation() -> None:
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    texts = [clean_text(item["text"]) for item in test_data]
    true_labels = [item["intent"] for item in test_data]

    classifier = IntentClassifier()
    classifier.load(CLASSIFIER_PATH, LABEL_ENCODER_PATH, CENTROIDS_PATH)

    predicted_labels = []
    confidences = []
    for text in texts:
        intent, confidence = classifier.predict(text)
        predicted_labels.append(intent)
        confidences.append(confidence)

    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, zero_division=0)
    matrix = confusion_matrix(true_labels, predicted_labels)

    avg_confidence = sum(confidences) / len(confidences)

    output_lines = [
        "=== Intent Classifier Evaluation Report ===",
        f"Test samples: {len(test_data)}",
        f"Accuracy: {accuracy:.2%}",
        f"Average confidence: {avg_confidence:.2%}",
        "",
        "Classification Report:",
        report,
        "Confusion Matrix (rows=true, cols=predicted):",
        str(matrix),
    ]
    report_text = "\n".join(output_lines)

    print(report_text)

    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nFull report saved to: {EVAL_REPORT_PATH}")


if __name__ == "__main__":
    run_evaluation()
