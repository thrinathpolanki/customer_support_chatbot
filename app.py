"""
Streamlit chat UI for demonstrating the chatbot.
Uses the ChatbotEngine directly (in-process) so the demo works with a
single command — no need to run the FastAPI server separately.

Run with:
    streamlit run app.py
"""

import uuid

import streamlit as st

from src.chatbot_engine import ChatbotEngine

st.set_page_config(
    page_title="AI Customer Support Chatbot",
    page_icon="💬",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading AI models (first run only)...")
def load_engine() -> ChatbotEngine:
    return ChatbotEngine()


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    init_session_state()
    engine = load_engine()

    st.title("💬 Intelligent Customer Support Chatbot")
    st.caption(
        "Hybrid AI: sentence-transformer intent classification with a "
        "FLAN-T5 generative fallback for out-of-scope questions."
    )

    with st.sidebar:
        st.header("Session Info")
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")

        if st.button("🔄 Reset Conversation"):
            engine.reset_session(st.session_state.session_id)
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.header("How it works")
        st.markdown(
            """
            1. Your message is embedded using a sentence transformer.
            2. A logistic regression model + cosine-similarity check
               predicts the **intent** and a calibrated confidence.
            3. High confidence → curated response.
            4. Low confidence → generative (FLAN-T5) fallback,
               using the last exchange as context.
            """
        )

    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                badge = "🧠 Generative fallback" if meta["used_fallback"] else "🎯 Intent match"
                st.caption(
                    f"{badge} — intent: `{meta['intent']}` "
                    f"(confidence: {meta['confidence']:.0%})"
                )

    # Chat input
    user_input = st.chat_input("Ask about your order, shipping, returns, refunds...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = engine.process_message(
                    st.session_state.session_id, user_input
                )
            st.markdown(result["response"])
            badge = "🧠 Generative fallback" if result["used_fallback"] else "🎯 Intent match"
            st.caption(
                f"{badge} — intent: `{result['intent']}` "
                f"(confidence: {result['confidence']:.0%})"
            )
            if result["escalate_to_human"]:
                st.warning("This conversation has been flagged for human escalation.")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["response"],
                "meta": result,
            }
        )


if __name__ == "__main__":
    main()
