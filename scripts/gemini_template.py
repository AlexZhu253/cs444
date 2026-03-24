"""
Template script for Google Gemini API.
Requires: pip install google-genai python-dotenv
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # loads variables from .env in the project root

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def chat(prompt: str, model: str = "gemini-2.0-flash", system: str = "You are a helpful assistant.") -> str:
    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(system_instruction=system),
        contents=prompt,
    )
    return response.text


if __name__ == "__main__":
    user_prompt = "Explain the difference between supervised and unsupervised learning in one paragraph."
    print(chat(user_prompt))
