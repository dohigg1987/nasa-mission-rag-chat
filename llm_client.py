"""
LLM client for the NASA mission chat system.

Sends the user's question, the retrieved mission context and a trimmed
conversation history to an OpenAI chat model and returns the answer.
"""

import os
from typing import Dict, List, Optional

from openai import OpenAI

# Vocareum (Udacity) keys must be sent to the Vocareum OpenAI proxy.
VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"

# Only the most recent turns are sent to the model. Older turns rarely help a
# factual question and they cost tokens, so the history is capped.
MAX_HISTORY_MESSAGES = 6

SYSTEM_PROMPT = """You are a NASA mission operations expert and historian specialising in the \
Apollo 11, Apollo 13 and Space Shuttle Challenger (STS-51-L) missions.

You answer questions using the retrieved excerpts from NASA mission transcripts and \
technical documents that are supplied to you as context.

Rules:
1. Base every factual statement on the supplied context. Do not add facts that are not \
supported by it.
2. Cite the excerpts you rely on using their labels, for example [Source 1] or [Source 2].
3. If the context does not contain enough information to answer, say so clearly \
(for example: "The retrieved documents do not cover this.") and, if helpful, state what \
is missing instead of guessing.
4. When transcripts are ambiguous or partially garbled (they come from scanned and \
transcribed records), say that the record is unclear rather than inventing details.
5. Be precise with names, times (including mission elapsed time), and technical terms, \
and keep answers concise and well structured."""


def resolve_base_url(api_key: str) -> Optional[str]:
    """Return the API base URL to use for a given key.

    An explicit OPENAI_BASE_URL environment variable always wins. Otherwise
    Udacity/Vocareum keys (which start with "voc-") are routed to the
    Vocareum proxy and standard OpenAI keys use the default endpoint.
    """
    env_url = os.getenv("OPENAI_BASE_URL")
    if env_url:
        return env_url
    if api_key and api_key.startswith("voc-"):
        return VOCAREUM_BASE_URL
    return None


def build_messages(user_message: str, context: str,
                   conversation_history: List[Dict]) -> List[Dict[str, str]]:
    """Assemble the message list: system prompt, context, recent history, question."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Context is supplied as its own system message so the model can tell
    # retrieved evidence apart from the conversation.
    if context and context.strip():
        messages.append({
            "role": "system",
            "content": "Retrieved NASA document context for the current question:\n\n" + context,
        })
    else:
        messages.append({
            "role": "system",
            "content": "No relevant NASA documents were retrieved for the current question. "
                       "Tell the user that the archive does not appear to cover it rather than "
                       "answering from general knowledge.",
        })

    # Keep only well-formed user/assistant turns, and only the most recent ones.
    clean_history = [
        {"role": turn["role"], "content": str(turn["content"])}
        for turn in (conversation_history or [])
        if isinstance(turn, dict)
        and turn.get("role") in ("user", "assistant")
        and turn.get("content")
    ]
    messages.extend(clean_history[-MAX_HISTORY_MESSAGES:])

    messages.append({"role": "user", "content": user_message})
    return messages


def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""
    if not openai_key:
        raise ValueError("An OpenAI API key is required.")
    if not user_message or not user_message.strip():
        raise ValueError("The question is empty.")

    messages = build_messages(user_message, context, conversation_history)

    client = OpenAI(api_key=openai_key, base_url=resolve_base_url(openai_key))

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,   # low temperature keeps answers close to the evidence
        max_tokens=800,
    )

    return response.choices[0].message.content.strip()
