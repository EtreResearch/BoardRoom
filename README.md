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

- `--tui` — launch a full-screen Textual UI (chat log + agent roster + live tally).
- `--rounds N` — number of discussion rounds before the verdict (default 3).
- `--model ID` — override the model for all agents (e.g. `claude-haiku-4-5-20251001`).
- `--config PATH` — use a different agents file.
- `--ordered` — speak in YAML order every round instead of shuffling.
- `--seed N` — seed the per-round shuffle for reproducibility.
- `--no-setup` — skip the TUI setup screen (use the flag values directly).

## TUI mode

```bash
python boardroom.py --tui "An AI app that generates personalized bedtime stories for kids."
```

A polished interface opens in your terminal:

- **Setup screen** — a brief picker at startup chooses the speaking order
  for the run (Shuffled / Structured). ← / → toggles, Enter accepts.
  Skipped if you pass `--no-setup`.
- **Header** — title and idea.
- **Chat log** (left) — finalized turns scroll by; the in-flight turn streams
  into a panel pinned at the bottom. Each round header shows the speaking
  order picked for that round.
- **Sidebar** (right) — each agent's live status (`pending` → `speaking…` →
  `done` / `voted GOOD|BAD`) and a running tally.
- **Footer** — key bindings.

Key bindings:
- `q` — quit
- `s` — save the transcript so far to `transcript-YYYYMMDD-HHMMSS.md`

## Customizing the executives

Edit `agents.yaml`. Each agent has a `role`, `color` (any rich color name),
and a `system` prompt that defines their lens. Add or remove agents freely —
the script handles any number.

## How it works

For each round, every agent receives the running transcript and speaks once.
By default the speaking order is **shuffled per round** so no single executive
always frames the discussion or always speaks last (the first speaker gets
zero context; the last speaker reads three prior turns — fixed order quietly
biases the conversation). Pass `--ordered` to keep the YAML order, or
`--seed N` to reproduce a specific shuffle. Token output streams to the
terminal as it's generated. After the final round, each agent is asked for
a one-line `VERDICT: GOOD | BAD` plus reasoning; the script tallies and
prints the majority verdict.

Each agent's persona is sent with `cache_control: ephemeral` so it's cached
across turns, keeping cost down.
