# BoardRoom

Four virtual executives — CEO, CTO, Head Architect, Head of Product — debate
your business idea in your terminal and deliver a verdict. Each turn streams
live so you can watch the conversation play out.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY
```

## Run

```bash
python boardroom.py "An AI app that generates personalized bedtime stories for kids."
```

Optional flags:

- `--rounds N` — number of discussion rounds before the verdict (default 3).
- `--model ID` — override the model for all agents (e.g. `claude-haiku-4-5-20251001`).
- `--config PATH` — use a different agents file.

## Customizing the executives

Edit `agents.yaml`. Each agent has a `role`, `color` (any rich color name),
and a `system` prompt that defines their lens. Add or remove agents freely —
the script handles any number.

## How it works

For each round, every agent receives the running transcript and speaks once,
in order. Token output streams to the terminal as it's generated. After the
final round, each agent is asked for a one-line `VERDICT: GOOD | BAD` plus
reasoning; the script tallies and prints the majority verdict.

Each agent's persona is sent with `cache_control: ephemeral` so it's cached
across turns, keeping cost down.
