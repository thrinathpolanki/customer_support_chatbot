"""
Handles response generation for both tiers of the chatbot:

1. IntentResponseSelector — Tier 1, picks a curated, business-approved
   response for a known intent and fills in any slots (like order IDs).

2. FallbackGenerator — Tier 2, uses a small instruction-tuned transformer
   (FLAN-T5) to generate a free-form response when the intent classifier
   isn't confident, using recent conversation history as context.

FIX NOTES (read this if you hit the "bot repeats my prompt back" bug):
Small instruction-tuned models like flan-t5-small can degenerate into
copying the input prompt almost verbatim when given a long, complex,
multi-part instruction (e.g. "You are a helpful assistant... Conversation
so far: ... Customer's latest message: ... Assistant's response:"). Two
things fix this:
  1. Use a SHORT, simple prompt format (T5 models were instruction-tuned
     mostly on short, direct tasks -- not long role-play scripts).
  2. Use beam search (not greedy decoding) with `no_repeat_ngram_size`,
     which strongly discourages the model from copying long spans of
     the input verbatim.
  3. As a final safety net, detect if the output overlaps too heavily
     with the input prompt and swap in a safe canned response instead
     of ever showing the user a broken/echoed reply.
"""

import json
import random

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import (
    GENERATIVE_MODEL_NAME,
    MAX_GENERATED_TOKENS,
    MIN_GENERATED_TOKENS,
    SAFE_FALLBACK_MESSAGE,
)


class IntentResponseSelector:
    def __init__(self, intents_path: str):
        with open(intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build a lookup: intent tag -> list of candidate responses
        self.responses_by_tag: dict[str, list[str]] = {
            intent["tag"]: intent["responses"] for intent in data["intents"]
        }

    def select(self, tag: str, slots: dict | None = None) -> str:
        """
        Picks a random response template for the given intent and fills
        in any {slot} placeholders (e.g. {order_id}) using the provided
        slots dictionary. Missing slots default gracefully.
        """
        slots = slots or {}
        candidates = self.responses_by_tag.get(tag)

        if not candidates:
            return SAFE_FALLBACK_MESSAGE

        template = random.choice(candidates)
        try:
            return template.format(**slots)
        except KeyError:
            # If the template needs a slot we don't have, fall back to
            # the raw template with the placeholder defaulted gracefully.
            return template.format(order_id="on file", **{
                k: v for k, v in slots.items() if k != "order_id"
            })


class FallbackGenerator:
    def __init__(self):
        # We load the tokenizer + model directly (instead of using
        # transformers' pipeline("text2text-generation", ...)) because the
        # set of named pipeline "tasks" has changed across transformers
        # versions, and "text2text-generation" is not guaranteed to exist
        # as a registered task string. AutoModelForSeq2SeqLM + .generate()
        # is the stable, version-proof way to run an encoder-decoder model
        # like FLAN-T5 regardless of which transformers release is installed.
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATIVE_MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(GENERATIVE_MODEL_NAME)
        self.model.eval()

    def generate(self, conversation_history: str, latest_message: str) -> str:
        """
        Builds a short, direct instruction prompt (short prompts work far
        more reliably with small T5 models than long role-play scripts),
        generates a response with beam search + repetition controls, and
        falls back to a safe canned message if the output looks broken
        or echoes the input.
        """
        # Only include the last exchange as context -- short prompts are
        # much less likely to cause small models to degenerate/echo.
        history_snippet = self._last_exchange(conversation_history)

        if history_snippet:
            prompt = (
                "Answer as a polite customer support agent in one or two "
                f"short sentences.\nPrevious message: {history_snippet}\n"
                f"Customer question: {latest_message}\nAnswer:"
            )
        else:
            prompt = (
                "Answer as a polite customer support agent in one or two "
                f"short sentences.\nCustomer question: {latest_message}\nAnswer:"
            )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_GENERATED_TOKENS,
                min_new_tokens=MIN_GENERATED_TOKENS,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
                early_stopping=True,
                do_sample=False,
            )

        generated_text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

        if self._is_degenerate(generated_text, prompt):
            return SAFE_FALLBACK_MESSAGE

        return generated_text

    @staticmethod
    def _last_exchange(history_text: str, max_lines: int = 2) -> str:
        """Keeps only the last `max_lines` lines of history for a tight prompt."""
        if not history_text:
            return ""
        lines = [line for line in history_text.split("\n") if line.strip()]
        return " ".join(lines[-max_lines:])

    @staticmethod
    def _is_degenerate(generated_text: str, prompt: str) -> bool:
        """
        Detects failure modes we never want to show a user:
        - Empty output
        - Output that is (near-)identical to the input prompt (echoing)
        - Output that just restates instruction boilerplate
        """
        if not generated_text or len(generated_text) < 2:
            return True

        normalized_output = generated_text.lower().strip()
        normalized_prompt = prompt.lower().strip()

        # Direct echo: model returned the prompt (or most of it) verbatim.
        if normalized_output in normalized_prompt:
            return True

        # High word-overlap with the prompt is a strong signal of copying
        # rather than genuine generation.
        prompt_words = set(normalized_prompt.split())
        output_words = normalized_output.split()
        if output_words:
            overlap_ratio = sum(1 for w in output_words if w in prompt_words) / len(output_words)
            if overlap_ratio > 0.85 and len(output_words) > 6:
                return True

        # Model literally repeating instruction scaffolding.
        boilerplate_markers = ["customer question:", "previous message:", "answer as a"]
        if any(marker in normalized_output for marker in boilerplate_markers):
            return True

        return False
