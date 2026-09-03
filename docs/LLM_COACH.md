# Viascope Coach setup

The `/explore` Coach works immediately without a paid model. In **rules mode** it explains the visible IPEDS/BLS evidence and asks a practical follow-up. Model access is an optional reasoning and writing layer; recommendations remain grounded in evidence supplied by Viascope.

## Free local setup with Ollama

Install Ollama, then run:

```bash
ollama pull qwen2.5:7b
ollama serve
```

Create `site/.env.local` (ignored by Git):

```dotenv
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b
```

In another terminal, run `cd site`, then `npm run dev`. Open `http://localhost:3000/explore`, choose an interest, open the Coach, and ask a question. An **AI-assisted** response label confirms the local model answered. Ollama on a laptop cannot serve the published website; production needs a hosted model endpoint.

## Optional hosted OpenAI setup

Add these server-only variables to `site/.env.local` locally, or to the hosting project's secret/environment settings in production:

```dotenv
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

Never commit `.env.local`, expose the key in browser code, or name it `NEXT_PUBLIC_OPENAI_API_KEY`. The server route calls the Responses API and the browser receives only the answer.

## Hugging Face Inference Providers

Create a fine-grained Hugging Face token with permission to call Inference Providers, then add this to `site/.env.local`:

```dotenv
HF_TOKEN=your_hugging_face_token
HF_MODEL=openai/gpt-oss-120b:fastest
```

Viascope uses Hugging Face's OpenAI-compatible Responses endpoint. Some accounts receive limited free inference credits; availability and cost depend on the selected model and provider. In provider order, Viascope tries Ollama, then Hugging Face, then OpenAI, and finally rules mode.

## Run the Coach evaluation suite

From `site/` run:

```bash
npm run eval:coach
```

The offline suite checks three release gates:

- **Factual grounding:** percentages in the answer must exist in the supplied evidence.
- **Usefulness:** the response must provide a comparison, verification step, or useful follow-up.
- **Safe confidence:** the response must avoid guarantees and acknowledge uncertainty or the need to verify.

`npm test` also contains adversarial tests that confirm invented numbers and guarantee language are rejected. Before changing a model or prompt, capture a larger set of representative questions and compare the old and new configurations against the same evaluation set.

## Failure behavior and data boundary

- No provider configured: rules mode.
- Ollama configured: local model first.
- OpenAI configured without Ollama: hosted model.
- Hugging Face configured without Ollama: hosted open-weight model.
- Provider unavailable or times out: automatic rules-mode fallback.
- Each request contains the current question, interest/work style, and no more than five visible field summaries. It does not contain the full dataset, precise geolocation, or an account identity.

Copy `.env.example` as a starting point. The committed file contains names and examples only—never secrets.
