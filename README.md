# BoardRoom

Four virtual executives — CEO, CTO, Head Architect, Head of Product — debate
your business idea in your terminal and deliver a verdict. Each turn streams
live so you can watch the conversation play out.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY
```

## Run

```bash
python boardroom.py --tui --rounds 2 "An AI app that generates personalized bedtime stories for politicians."
```

Optional flags:

- `--tui` — launch a full-screen Textual UI (chat log + agent roster + live tally).
- `--rounds N` — number of discussion rounds before the verdict (default 3).
- `--model ID` — override the model for all agents (e.g. `claude-haiku-4-5-20251001`).
- `--config PATH` — use a different agents file.
- `--ordered` — speak in YAML order every round instead of shuffling.
- `--seed N` — seed the per-round shuffle for reproducibility.
- `--no-setup` — skip the TUI setup screen (use the flag values directly).
- `--verdict {decision_frame,simple,scorecard,recommendation}` — verdict-round
  style. Default is `decision_frame` (confidence + steel-man + synthesized
  case-for/against). `simple` reverts to the original `N GOOD / M BAD →
  VERDICT` tally with no confidence collection and no synthesis call —
  useful for batch / scripted runs that want one number per idea at
  minimum cost. `scorecard` swaps the GOOD/BAD vote for a per-dimension
  1-5 rating across Market, Tech, UX, and Risk, plus per-dimension
  averages and a composite score — useful when you want quantitative
  comparison across multiple ideas. `recommendation` asks each agent
  for an action (PROCEED / PAUSE / PIVOT / KILL) plus a 2-3 sentence
  rationale — useful when the question isn't "is this good" but "what
  should we do".

## TUI mode

```bash
python boardroom.py --tui "An AI app that generates personalized bedtime stories for politicians."
```
<img width="1476" height="937" alt="Screenshot 2026-05-21 at 14 58 45" src="https://github.com/user-attachments/assets/fcbfadcf-1199-4780-b7aa-00a2fea1ac8f" />

A polished interface opens in your terminal:

- **Setup screen** — a brief picker at startup chooses the speaking order
  for the run (Shuffled / Structured). ← / → toggles, Enter accepts.
  Skipped if you pass `--no-setup`.
- **Header** — title and idea.
- **Chat log** (left) — finalized turns scroll by; the in-flight turn streams
  into a panel pinned at the bottom. Each round header shows the speaking
  order picked for that round.
- **Sidebar** (right) — each agent's live status (`pending` → `speaking…` →
  `done` / `voted GOOD|BAD`), a running tally, and a **Usage** panel that
  shows cumulative input/output/cache tokens and an estimated dollar cost,
  updating after every turn.
- **Footer** — key bindings.

Key bindings:
- `q` — quit
- `s` — save the transcript so far to `transcript-YYYYMMDD-HHMMSS.md`
- `i` — interject. Opens a modal where you can type a one-shot directive
  (e.g. "focus on privacy concerns", "what about the European market?")
  to steer the discussion. The current round keeps streaming uninterrupted;
  your directive is queued and attached to every agent's prompt in the
  *next* round (or the verdict round if you queued during the final round).

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
across turns, keeping cost down. Per-turn token usage from the Anthropic API
is surfaced to the UI: the TUI's sidebar updates after every turn, and
stdout mode prints a one-line `Usage:` summary at the end. Cost estimates
use a hardcoded price table in `engine.MODEL_PRICING` — edit that dict if
Anthropic's published pricing changes.

## The verdict tally

The verdict round asks each agent for three things, not just one: their vote
(`GOOD` / `BAD`), a **confidence** score from 1 to 5, and one or two
sentences of reasoning. The tally then surfaces conviction strength
alongside the raw count so a borderline GOOD can't visually masquerade as a
strongly held one:

```
Verdict: moderate GOOD  ·  ⚠ 1 strong dissent
GOOD  1 strong · 2 lean
BAD   1 strong · 0 lean
Weighted: +35%  (raw: 3 GOOD / 1 BAD)
```

- **Confidence buckets**: 4–5 → *strong*, 2–3 → *lean*, 1 (or missing) → *weak*.
- **Headline** combines direction and conviction: *weak / moderate / strong GOOD*
  or *BAD*, or *split / deeply split* for ties.
- **⚠ Strong dissent** appears when any agent voted opposite the majority
  with confidence ≥ 4 — the dissenter most worth listening to is parked
  right next to the headline.
- **Weighted** is the signed conviction in [−100%, +100%] (sum of
  `confidence × sign` divided by maximum possible).

After the tally, a one-shot synthesis call produces a **decision frame** —
the panel's neutral one-page summary:

- **Case for** — two-sentence affirmative synthesis across the GOOD voters.
- **Case against** — two-sentence objection, surfaced honestly even if it's
  a minority view. High-confidence dissent shows up here.
- **Biggest unknown** — the single question whose answer would most change
  the panel's view.
- **Conditions for proceeding** — concrete preconditions the idea depends on.

Each agent's verdict prompt also asks for their own **strongest counter-argument**
(the "what would change your mind" steel-man), which appears in the saved
transcript next to their vote. The synthesis call costs roughly one extra
agent-turn (~$0.005–$0.01 on Sonnet) per run.
