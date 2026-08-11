"""
The chatbot engine orchestrates the full pipeline:

  user message
      -> preprocessing
      -> intent classification (Tier 1)
      -> confidence check
          -> high confidence: curated templated response
          -> low confidence: generative fallback (Tier 2), context-aware
      -> context update
      -> structured response returned to caller (API / UI)

This is the single object both api.py and app.py depend on, so business
logic lives in exactly one place.
"""

from src.config import (
    CENTROIDS_PATH,
    CLASSIFIER_PATH,
    CONFIDENCE_THRESHOLD,
    FALLBACK_TAG,
    INTENTS_PATH,
    LABEL_ENCODER_PATH,
    MAX_CONTEXT_TURNS,
)
from src.context_manager import ContextManager
from src.intent_classifier import IntentClassifier
from src.preprocessing import clean_text, extract_order_id
from src.response_generator import FallbackGenerator, IntentResponseSelector


class ChatbotEngine:
    def __init__(self):
        # --- Load Tier 1: intent classifier ---
        self.classifier = IntentClassifier()
        self.classifier.load(CLASSIFIER_PATH, LABEL_ENCODER_PATH, CENTROIDS_PATH)

        # --- Load response systems ---
        self.response_selector = IntentResponseSelector(INTENTS_PATH)
        self.fallback_generator = FallbackGenerator()

        # --- Context memory ---
        self.context_manager = ContextManager(max_turns=MAX_CONTEXT_TURNS)

        self.confidence_threshold = CONFIDENCE_THRESHOLD

    def process_message(self, session_id: str, message: str) -> dict:
        """
        Processes a single user message end-to-end and returns a
        structured result describing what happened, so the frontend
        can show intent/confidence for transparency (great for demos).
        """
        context = self.context_manager.get_or_create(session_id)
        context.add_user_message(message)

        cleaned_message = clean_text(message)
        order_id = extract_order_id(message)

        intent, confidence = self.classifier.predict(cleaned_message)

        used_fallback = False
        escalate = False

        if confidence >= self.confidence_threshold:
            # Tier 1: confident, curated response
            response_text = self.response_selector.select(
                intent, slots={"order_id": order_id}
            )
            if intent == "human_agent":
                escalate = True
        else:
            # Tier 2: not confident enough, use generative fallback
            used_fallback = True
            intent = FALLBACK_TAG
            history_text = context.get_history_text(max_turns=2)
            response_text = self.fallback_generator.generate(
                history_text, message
            )

        context.add_bot_message(response_text)
        context.last_intent = intent

        return {
            "session_id": session_id,
            "intent": intent,
            "confidence": round(confidence, 4),
            "response": response_text,
            "used_fallback": used_fallback,
            "escalate_to_human": escalate,
        }

    def reset_session(self, session_id: str) -> None:
        self.context_manager.reset_session(session_id)
