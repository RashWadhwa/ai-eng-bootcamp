"""Simple OpenAI tracing demo for the AI Evals session.

Uses the Langfuse OpenAI wrapper for automatic tracing and loads a prompt from
Langfuse Prompt Management.
"""

import os

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import openai


load_dotenv()

langfuse = get_client()
openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = langfuse.get_prompt("research_assistant_system_prompt")
prompt_text = prompt.prompt
model = prompt.config.get("model", "gpt-3.5-turbo") if prompt.config else "gpt-3.5-turbo"
temperature = prompt.config.get("temperature", 0.7) if prompt.config else 0.7

result = openai.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": "What are the key characteristics of Baroque music?"},
    ],
    temperature=temperature,
)

print("=" * 60)
print("OpenAI Response:")
print("=" * 60)
print(result.choices[0].message.content)
print("=" * 60)

langfuse.flush()

print("\nTrace sent to Langfuse. Check the dashboard in 2-5 seconds.")
print("Visit: https://cloud.langfuse.com")
