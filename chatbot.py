"""
chatbot.py — Sanjeevni's context-aware follow-up assistant.

Uses a SEPARATE Gemini API key/project from the main report-generation
key (GEMINI_CHATBOT_API_KEY, not GEMINI_API_KEY), so a chatbot
conversation's request volume can never compete with or starve
report-generation quota, and vice versa.

Defaults to gemini-3.1-flash-lite -- answering follow-ups grounded in an
ALREADY-generated report is a lighter task than the original multi-source
synthesis, and Flash-Lite's higher free-tier headroom (500 RPD / 15 RPM,
vs. gemini-3-flash-preview's 20 RPD) comfortably covers real
conversational back-and-forth.

Uses the Interactions API's previous_interaction_id for server-side
multi-turn state (confirmed against Google's current docs) -- only the
NEW message is sent each turn; the server keeps the actual conversation
history, so the full report is never resent after the first message.

Deliberately self-contained -- does NOT import anything from
gemini_client.py, even though some logic (timeout wrapper, response
parsing) is duplicated rather than shared. This keeps the chatbot and
report-generation paths fully independent, matching the reason they were
split onto separate API keys/projects in the first place.
"""

import os
import json
import concurrent.futures
from google import genai

CHATBOT_MODEL_NAME = os.environ.get("SANJEEVNI_CHATBOT_MODEL", "gemini-3.1-flash-lite")
CHATBOT_TIMEOUT_SECONDS = 60  # chatbot turns are lighter/shorter than a full report generation, so a shorter timeout is fine here

chatbot_api_key = os.environ.get("GEMINI_CHATBOT_API_KEY")
assert chatbot_api_key, "GEMINI_CHATBOT_API_KEY not found in environment -- setx it under the chatbot's own Google Cloud project"
chatbot_client = genai.Client(api_key=chatbot_api_key)


def _call_with_timeout(fn, *args, timeout_seconds=CHATBOT_TIMEOUT_SECONDS, **kwargs):
    """Same timeout trick I use in gemini_client.py, duplicated here on purpose
    instead of imported, so this file doesn't end up depending on that one at
    all. Not relying on the SDK's own timeout parameter alone -- it's been
    silently ignored on some request paths before."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"The assistant did not respond within {timeout_seconds} seconds. "
                f"This is usually a transient network issue -- please try again."
            )


def get_interaction_text(interaction) -> str:
    """Same defensive parsing as gemini_client.py -- Google's own docs have
    shown Interactions API responses coming back in two slightly different
    shapes, so checking for both here."""
    text = getattr(interaction, "output_text", None)
    if text:
        return text
    for step in getattr(interaction, "steps", []):
        if getattr(step, "type", None) == "model_output":
            for block in getattr(step, "content", []):
                if getattr(block, "type", None) == "text":
                    return block.text
    raise ValueError("Could not extract text from interaction response -- inspect interaction object directly")


def build_seed_context(farmer_report: dict) -> str:
    """Only used for the very first message in a conversation -- seeds the
    assistant with the farmer's full report, so follow-up answers stay
    consistent with what was already recommended instead of starting blank
    or contradicting the report."""
    return f"""You are Sanjeevni's follow-up assistant. A smallholder farmer has just
received the crop recommendation report below.

Ground your answers in THIS report wherever it's relevant -- don't contradict it, and
never invent specific numbers, probabilities, or figures that would falsely appear to
come from their exact soil test or location analysis (never make up a yield figure,
price, or soil-fit score that isn't already in the report).

However, farmers will naturally ask practical questions that go beyond the report's
specific scope -- things like farm machinery and equipment options, pest and disease
management, where to source seeds or fertilizer, or general cultivation practices. For
these, DO help using your own general agricultural knowledge -- refusing a genuinely
useful, reasonable question undermines the farmer's trust in this assistant just as much
as being wrong would. Just be clear about which is which: say plainly when you're
drawing on general farming knowledge rather than their specific report (for example,
"this isn't from your specific soil report, but generally speaking...") so they
understand what's tailored to their exact situation versus general advice.

Only decline to answer when something is genuinely impossible to know (real-time
weather, today's exact market price, a specific named product/brand you have no
reliable knowledge of) -- and even then, offer whatever general guidance you reasonably
can alongside the honest caveat, rather than refusing outright.

Keep every answer in simple, plain English -- short sentences, everyday words, the
same easy reading level as the report itself. Avoid technical jargon.

THE FARMER'S FULL REPORT:
{json.dumps(farmer_report, indent=2)}

The farmer's question:
"""


def send_chat_message(user_message: str, farmer_report: dict, previous_interaction_id: str = None):
    """
    Sends one chatbot turn. Returns (response_text, new_interaction_id).

    First call in a conversation (previous_interaction_id is None): seeds
    the full report as context alongside the farmer's first question.
    Every call after that: just the new message -- store=True plus
    previous_interaction_id means the server already has the full history,
    so the report never needs to be sent again.
    """
    if previous_interaction_id is None:
        full_input = build_seed_context(farmer_report) + user_message
    else:
        full_input = user_message

    kwargs = {
        "model": CHATBOT_MODEL_NAME,
        "input": full_input,
        "store": True,
    }
    if previous_interaction_id:
        kwargs["previous_interaction_id"] = previous_interaction_id

    interaction = _call_with_timeout(chatbot_client.interactions.create, **kwargs)
    return get_interaction_text(interaction), interaction.id
