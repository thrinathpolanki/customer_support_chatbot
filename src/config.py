"""
Central configuration for the chatbot system.
Keeping all constants in one place makes the system easy to tune
without hunting through business logic files.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

INTENTS_PATH = os.path.join(DATA_DIR, "intents.json")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.json")

CLASSIFIER_PATH = os.path.join(MODELS_DIR, "intent_classifier.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
CENTROIDS_PATH = os.path.join(MODELS_DIR, "intent_centroids.joblib")
EVAL_REPORT_PATH = os.path.join(MODELS_DIR, "evaluation_report.txt")

# --- Models ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"       # Sentence-transformer for intent embeddings

# google/flan-t5-base gives noticeably more coherent fallback replies than
# flan-t5-small (which tends to degenerate/echo the prompt on complex
# instructions). It's still CPU-friendly (~250M params). If you're on a
# very slow machine you can switch this back to "google/flan-t5-small",
# but expect lower-quality fallback answers.
GENERATIVE_MODEL_NAME = "google/flan-t5-base"

# --- Chatbot behavior ---
# NOTE on this threshold: confidence is computed as the MAX of (a) the
# logistic regression's predicted probability and (b) cosine similarity
# to the nearest intent's example embeddings (see intent_classifier.py).
# This blended score is much better calibrated than raw softmax
# probability alone -- a clean match like "hi" -> greeting will score
# ~0.85-1.0 instead of the ~0.3-0.5 you'd see from softmax probability
# spread thinly across 11 classes. 0.45 is a safe, tested default.
CONFIDENCE_THRESHOLD = 0.45
MAX_CONTEXT_TURNS = 6         # Number of past turns kept in memory per session
MAX_GENERATED_TOKENS = 60     # Cap on fallback response length
MIN_GENERATED_TOKENS = 4      # Floor, avoids empty generations

# Intents that should never trigger the generative fallback even at low
# confidence risk (kept simple here — extendable per business rules).
FALLBACK_TAG = "fallback"

# Safe, canned message used when the generative fallback fails or
# produces a degenerate/echoed response (see response_generator.py).
SAFE_FALLBACK_MESSAGE = (
    "I'm not fully sure how to answer that one accurately. "
    "I don't want to guess and give you the wrong information — "
    "would you like me to connect you with a human support agent?"
)
