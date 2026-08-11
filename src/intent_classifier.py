"""
Intent classification using sentence-transformer embeddings + logistic
regression. This is the "Tier 1" fast path of the chatbot.

Why this design:
- Sentence-transformers give us semantic understanding (so "where's my
  package" and "track my order" map to similar vectors) without needing
  to fine-tune a full transformer.
- Logistic regression on top is fast, interpretable, and gives calibrated
  probabilities via predict_proba(), which we use as a confidence score.

CONFIDENCE FIX:
With a small training set spread across 11 classes, raw softmax
probability from logistic regression is often under-confident even on
clean, obvious matches (e.g. "hi" -> greeting might score only ~0.3-0.4
because probability mass gets spread across many classes). To fix this,
we ALSO compute cosine similarity between the input and each intent's
average ("centroid") example embedding, and take the confidence as the
MAX of the LR probability and the top cosine similarity. This mirrors
how production semantic-search-based intent matchers actually gauge
confidence, and produces much better-calibrated scores.
"""

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.config import EMBEDDING_MODEL_NAME


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalizes rows so dot product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    return vectors / norms


class IntentClassifier:
    def __init__(self):
        # Loaded once and reused for both training and inference.
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.classifier: LogisticRegression | None = None
        self.label_encoder: LabelEncoder | None = None
        # tag -> normalized centroid embedding (mean of that intent's
        # training example embeddings)
        self.centroids: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Converts a list of raw strings into dense embedding vectors."""
        return self.embedder.encode(texts, show_progress_bar=False)

    def fit(self, texts: list[str], labels: list[str]) -> None:
        """
        Trains the classifier on (text, intent_label) pairs.
        Embeddings are computed once, then a logistic regression model
        learns to separate the intent classes in embedding space. We
        also compute a per-intent centroid embedding for similarity-based
        confidence boosting at inference time.
        """
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)

        embeddings = self.embed(texts)
        normalized_embeddings = _normalize(embeddings)

        # NOTE: `multi_class` was removed in scikit-learn 1.7+ (multinomial
        # is now used automatically by solvers that support it, like the
        # default "lbfgs"), so it's intentionally omitted here.
        self.classifier = LogisticRegression(
            max_iter=2000,
            C=3.0,  # less regularization -> sharper, more confident boundaries
            class_weight="balanced",
        )
        self.classifier.fit(embeddings, encoded_labels)

        # Build one centroid per intent tag from its normalized example embeddings.
        self.centroids = {}
        labels_array = np.array(labels)
        for tag in sorted(set(labels)):
            mask = labels_array == tag
            tag_embeddings = normalized_embeddings[mask]
            centroid = tag_embeddings.mean(axis=0)
            self.centroids[tag] = centroid / (np.linalg.norm(centroid) + 1e-8)

    def predict(self, text: str) -> tuple[str, float]:
        """
        Predicts the intent for a single piece of text.
        Returns (intent_label, confidence_score).

        confidence_score = max(logistic-regression probability for the
        predicted class, cosine similarity to the nearest intent centroid
        for that SAME class). This blended signal fixes the common issue
        where softmax probability alone under-reports confidence on
        clean, obvious matches.
        """
        if self.classifier is None or self.label_encoder is None:
            raise RuntimeError(
                "Classifier not trained or loaded. Call fit() or load() first."
            )

        embedding = self.embed([text])
        normalized = _normalize(embedding)[0]

        probabilities = self.classifier.predict_proba(embedding)[0]
        top_index = int(np.argmax(probabilities))
        lr_confidence = float(probabilities[top_index])
        intent = self.label_encoder.inverse_transform([top_index])[0]

        # Cosine similarity to this predicted intent's centroid.
        centroid = self.centroids.get(intent)
        cosine_confidence = float(np.dot(normalized, centroid)) if centroid is not None else 0.0
        # Cosine similarity ranges roughly [-1, 1]; clip negative noise to 0
        cosine_confidence = max(cosine_confidence, 0.0)

        confidence = max(lr_confidence, cosine_confidence)

        return intent, confidence

    def save(self, classifier_path: str, label_encoder_path: str, centroids_path: str) -> None:
        """Persists the trained classifier, label encoder, and centroids to disk."""
        joblib.dump(self.classifier, classifier_path)
        joblib.dump(self.label_encoder, label_encoder_path)
        joblib.dump(self.centroids, centroids_path)

    def load(self, classifier_path: str, label_encoder_path: str, centroids_path: str) -> None:
        """Loads a previously trained classifier, label encoder, and centroids."""
        self.classifier = joblib.load(classifier_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.centroids = joblib.load(centroids_path)
