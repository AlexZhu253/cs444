"""
Template script for DeepSeek API.
DeepSeek is OpenAI-compatible, so we reuse the openai SDK with a custom base_url.
Requires: pip install openai python-dotenv
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads variables from .env in the project root

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


def chat(prompt: str, model: str = "deepseek-chat", system: str = "You are a helpful assistant.") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    user_prompt = "Explain the difference between supervised and unsupervised learning in one paragraph."
    print(chat(user_prompt))
