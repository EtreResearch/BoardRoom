# BoardRoom

Four virtual executives — CEO, CTO, Head Architect, Head of Product — debate
your business idea in your terminal and deliver a verdict. Each turn streams
live so you can watch the conversation play out.

Bring your own LLM: Anthropic Claude, OpenAI, or anything OpenAI-compatible
— including local servers like **Ollama**, LM Studio, vLLM, llama.cpp, and
LocalAI. Local options are free.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then pick a provider with the interactive wizard:

```bash
python boardroom.py init
```

The wizard lists providers (local options first), pings the one you pick,
shows the available models, and saves your choice into `agents.yaml`.

### Quickstart with Ollama (free, local)

```bash
ollama serve                      # in one terminal
ollama pull llama3.1:8b           # any chat model works
python boardroom.py init          # pick "ollama", pick the model
python boardroom.py "An AI app that generates personalized bedtime stories for kids."
```

### Quickstart with Anthropic Claude (paid, cloud)

```bash
cp .env.example .env              # then set ANTHROPIC_API_KEY
python boardroom.py init          # pick "anthropic", pick a model
python boardroom.py "..."
```

## Providers

| Key          | Where it runs       | Cost                | API key                          |
|--------------|---------------------|---------------------|----------------------------------|
| `ollama`     | localhost:11434     | free                | none                             |
| `lmstudio`   | localhost:1234      | free                | none                             |
| `vllm`       | localhost:8000      | free (self-hosted)  | none                             |
| `localai`    | localhost:8080      | free (self-hosted)  | none                             |
| `anthropic`  | cloud               | per-token           | `ANTHROPIC_API_KEY`              |
| `openai`     | cloud               | per-token           | `OPENAI_API_KEY`                 |
| `openrouter` | cloud               | per-token           | `OPENROUTER_API_KEY`             |
| `groq`       | cloud               | free tier available | `GROQ_API_KEY`                   |
| `together`   | cloud               | per-token           | `TOGETHER_API_KEY`               |
| `custom`     | your URL            | depends             | depends                          |

## Run

```bash
python boardroom.py "An AI app that generates personalized bedtime stories for kids."
```

Optional flags:

- `--tui` — launch the full-screen focus-mode UI.
- `--rounds N` — number of discussion rounds before the verdict (default 3).
- `--provider KEY` — override the provider for one run (e.g. `--provider ollama`).
- `--model ID` — override the model.
- `--base-url URL` — override the OpenAI-compatible endpoint URL.
- `--api-key-env NAME` — env var that holds the API key.
- `--config PATH` — use a different `agents.yaml`.

## TUI mode

```bash
python boardroom.py --tui "An AI app that generates personalized bedtime stories for kids."
```

The interface runs in **focus mode** — calm and one-at-a-time:

- **Header** — title and idea
- **Sub-header** — current phase (`Round 2 of 3`, `Final verdicts`, `Decision`)
- **Focal card** — a single large centered card showing only the current
  speaker. The border picks up that agent's color and their words stream
  in live. When they finish, the next speaker fades in.
- **Status row** — a single compact line at the bottom showing every
  agent's state (○ pending · ◐ speaking · ● done · ✓ GOOD · ✗ BAD) plus
  the running tally. When the discussion ends, the focal card transforms
  into a final verdict banner.
- **Footer** — key bindings

Key bindings:
- `q` — quit
- `h` — toggle the history overlay (scrollable list of every past turn;
  the live discussion continues underneath, press `h` or `esc` to return)
- `s` — save the transcript so far to `transcript-YYYYMMDD-HHMMSS.md`

## Customizing the executives

Edit `agents.yaml`. Each agent has a `role`, `color` (any rich color name),
and a `system` prompt that defines their lens. Add or remove agents freely —
the script handles any number.

## How it works

For each round, every agent receives the running transcript and speaks once,
in order. Token output streams to the terminal as it's generated. After the
final round, each agent is asked for a one-line `VERDICT: GOOD | BAD` plus
reasoning; the script tallies and prints the majority verdict.

When the Anthropic provider is in use, each agent's persona is sent with
`cache_control: ephemeral` so it's cached across turns, keeping cost down.
Other providers fall back to plain system prompts.
