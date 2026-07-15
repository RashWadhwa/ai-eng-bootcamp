# AI Builders: AI Evals Session

This session introduces practical LLM evaluation and observability workflows using
OpenAI and Langfuse. Langfuse is the tracing and prompt-management tool in the demo,
but the session is organized around AI evals rather than as a standalone Langfuse module.

## What You Will Learn

- Trace OpenAI calls automatically with Langfuse.
- Load prompts from Langfuse Prompt Management.
- Send traces to a dashboard for inspection.
- Review latency, token usage, cost, prompt versions, and model responses.

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

Open `.env` and add your keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Get your keys:

- OpenAI: https://platform.openai.com/api-keys
- Langfuse: https://cloud.langfuse.com

### 3. Run the Demo

```bash
python openai_eval_tracing.py
```

## What the Script Does

1. Initializes a Langfuse client.
2. Loads `research_assistant_system_prompt` from Langfuse Prompt Management.
3. Calls OpenAI through the Langfuse OpenAI wrapper.
4. Flushes the trace so it appears in Langfuse quickly.

## Key Concepts

### Tracing

The Langfuse OpenAI wrapper traces model calls automatically:

```python
from langfuse.openai import openai

result = openai.chat.completions.create(...)
```

### Prompt Management

The script loads the prompt and optional config from Langfuse:

```python
prompt = langfuse.get_prompt("prompt-name")
prompt_text = prompt.prompt
model = prompt.config.get("model", "gpt-3.5-turbo")
```

### Eval Review

After running the script, open Langfuse and inspect:

- request and response payloads
- prompt version
- model, latency, tokens, and cost
- trace metadata that can support evaluation workflows

## Files

- `openai_eval_tracing.py`: simple OpenAI tracing demo
- `requirements.txt`: Python dependencies
- `.env.example`: environment variable template

## Related Examples

For a larger FastAPI and Streamlit example with tracing, scoring, and sessions, see
`ai-engineering-bootcamp/eval-monitoring-shipping/`.
