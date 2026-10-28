
import os
from contextvars import ContextVar

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

if not MODEL:
    raise ValueError("OPENROUTER_MODEL is missing from .env")


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)


# Track usage separately for each email-processing context.
_usage_context = ContextVar("llm_usage_context", default=None)


def start_usage_tracking():
    """Start a new usage-tracking context for one email."""
    return _usage_context.set({
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "usage_available": False,
        "cost_available": False,
    })


def get_usage() -> dict:
    """Return accumulated usage for the current context."""
    usage = _usage_context.get()

    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "estimated_cost_usd": None,
        }

    return {
        "input_tokens": (
            usage["input_tokens"]
            if usage["usage_available"] else None
        ),
        "output_tokens": (
            usage["output_tokens"]
            if usage["usage_available"] else None
        ),
        "estimated_cost_usd": (
            usage["estimated_cost_usd"]
            if usage["cost_available"] else None
        ),
    }


def stop_usage_tracking(token):
    """Restore the previous usage-tracking context."""
    _usage_context.reset(token)


def record_usage(response):
    """Accumulate token counts and reported provider cost."""
    context = _usage_context.get()

    if context is None or response.usage is None:
        return

    usage = response.usage

    context["input_tokens"] += usage.prompt_tokens or 0
    context["output_tokens"] += usage.completion_tokens or 0
    context["usage_available"] = True

    provider_cost = getattr(usage, "cost", None)

    if provider_cost is not None:
        context["estimated_cost_usd"] += float(provider_cost)
        context["cost_available"] = True


def ask_llm(prompt: str) -> str:
    """Send a prompt to OpenRouter and return the model response."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=1000,
    )

    # Record usage without changing the return type.
    record_usage(response)

    message = response.choices[0].message
    answer = message.content or ""

    if not answer.strip():
        print("Warning: The model returned an empty answer.")
        print("Model used:", response.model)
        print("Finish reason:", response.choices[0].finish_reason)
        print("Usage:", response.usage)

    return answer


if __name__ == "__main__":
    token = start_usage_tracking()

    try:
        result = ask_llm(
            "Classify this email in one sentence: "
            "My order has not arrived yet."
        )

        print("Model:", MODEL)
        print("Response:", result)
        print("Usage:", get_usage())

    finally:
        stop_usage_tracking(token)
