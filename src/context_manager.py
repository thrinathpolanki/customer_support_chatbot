"""
Manages per-session conversation memory so the bot has context across
multiple turns (e.g. remembering what the user was just asking about).

In production this would be backed by Redis or a database keyed by
session/user ID. For this project, an in-memory dictionary is used,
which is perfectly fine for a single-process demo/API server.
"""

from datetime import datetime


class ConversationContext:
    def __init__(self, session_id: str, max_turns: int = 6):
        self.session_id = session_id
        self.max_turns = max_turns
        self.history: list[dict] = []   # list of {"role", "text", "timestamp"}
        self.last_intent: str | None = None

    def add_user_message(self, text: str) -> None:
        self._add("user", text)

    def add_bot_message(self, text: str) -> None:
        self._add("bot", text)

    def _add(self, role: str, text: str) -> None:
        self.history.append(
            {
                "role": role,
                "text": text,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        # Keep only the most recent `max_turns` exchanges (user+bot pairs)
        max_entries = self.max_turns * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

    def get_history_text(self, max_turns: int | None = None) -> str:
        """
        Formats recent conversation history as plain text, used as
        context in the generative fallback prompt. `max_turns` lets
        callers request a shorter slice (small models do better with
        short, focused prompts).
        """
        entries = self.history
        if max_turns is not None:
            entries = entries[-(max_turns * 2):]

        lines = []
        for turn in entries:
            speaker = "Customer" if turn["role"] == "user" else "Agent"
            lines.append(f"{speaker}: {turn['text']}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.history = []
        self.last_intent = None


class ContextManager:
    """Holds one ConversationContext per active session."""

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.sessions: dict[str, ConversationContext] = {}

    def get_or_create(self, session_id: str) -> ConversationContext:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationContext(
                session_id, max_turns=self.max_turns
            )
        return self.sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            self.sessions[session_id].reset()
