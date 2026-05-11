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

Each agent's persona is sent with `cache_control: ephemeral` so it's cached
across turns, keeping cost down.
