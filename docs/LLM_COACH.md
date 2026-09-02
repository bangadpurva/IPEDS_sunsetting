# Optional LLM Coach

The app does not require an LLM. By default, AI Coach uses local rules plus the IPEDS/BLS dataset.

When configured, the server can call an LLM to write a more natural coaching response while keeping recommendations grounded in the local research data.

## Recommended Free Setup: Ollama

Use Ollama for a free local LLM. This avoids token billing and keeps prompts on your machine.

Install Ollama, then pull one model:

```bash
ollama pull qwen2.5:7b
```

Start the app with:

```bash
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export OLLAMA_MODEL="qwen2.5:7b"
python3 -B -m app.ipeds_connect.server
```

Good free model options:

| Model | Why use it |
|---|---|
| `qwen2.5:7b` | Good general reasoning, useful for skills and career-language interpretation |
| `llama3.1:8b` | Strong general conversation and explanation |
| `mistral:7b` | Fast responses on modest hardware |

For this app, a small local model is enough because the app supplies the IPEDS/BLS context and asks the model to explain, compare, and ask follow-up questions.

## OpenAI-Compatible API

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1-mini"
python3 -B -m app.ipeds_connect.server
```

Optional:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

`OPENAI_BASE_URL` can point to any OpenAI-compatible server.

## Local Ollama

```bash
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export OLLAMA_MODEL="qwen2.5:7b"
python3 -B -m app.ipeds_connect.server
```

## Behavior

- If an LLM is configured and responds, API results include `mode: "llm"`.
- If no LLM is configured, results include `mode: "rules"`.
- If the LLM call fails, the app falls back to rules and returns the failure reason.
- LLM output receives only a compact recommendation context, not the full raw dataset.

The LLM should explain choices, ask practical next-step questions, and flag when live job-market verification is needed.
