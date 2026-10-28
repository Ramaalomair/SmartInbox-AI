
import os

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


def ask_llm(prompt: str) -> str:
    """Send a prompt to OpenRouter and return the model's response."""

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

    message = response.choices[0].message
    answer = message.content or ""

    if not answer.strip():
        print("Warning: The model returned an empty answer.")
        print("Model used:", response.model)
        print("Finish reason:", response.choices[0].finish_reason)
        print("Usage:", response.usage)

    return answer


if __name__ == "__main__":
    result = ask_llm(
        "Classify this email in one sentence: "
        "My order has not arrived yet."
    )

    print("Model:", MODEL)
    print("Response:", result)
