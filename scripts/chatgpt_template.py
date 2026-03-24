"""
Template script for OpenAI ChatGPT API.
Requires: pip install openai python-dotenv
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads variables from .env in the project root

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def chat(prompt: str, model: str = "gpt-4o", system: str = "You are a helpful assistant.") -> str:
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
