import os

from dotenv import load_dotenv
from openai import OpenAI

from prompts import GENERATE_QUESITONS

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set in .env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def main() -> str:
    response = client.chat.completions.create(
        model="anthropic/claude-opus-4.6",
        messages=[{"role": "user", "content": GENERATE_QUESITONS}],
    )
    content = response.choices[0].message.content
    print(content)
    return content


if __name__ == "__main__":
    main()
