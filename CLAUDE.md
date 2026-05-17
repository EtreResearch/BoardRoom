# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install -r requirements.txt

# Configure: copy .env.example to .env and set ANTHROPIC_API_KEY

# Stdout mode (streams colored turns to the terminal)
python boardroom.py "<business idea>"

# Full-screen Textual UI
python boardroom.py --tui "<business idea>"

# Common overrides
python boardroom.py --rounds 1 "<idea>"
python boardroom.py --model claude-haiku-4-5-20251001 "<idea>"
python boardroom.py --config path/to/agents.yaml "<idea>"
```

There is no test suite or linter configured. For sanity-checking Python files
after edits, use `python -c "import ast; ast.parse(open('FILE').read())"`.
Validate the TUI by running it; type-checkers won't catch Textual layout bugs.

## Architecture

The codebase is small (~5 Python files) but its shape matters. It separates
the LLM-driven discussion logic from how that logic is shown to the user:

**Engine (`engine.py`)** — `run_boardroom(client, agents, idea, rounds, model, max_tokens)`
is an async generator yielding typed event dataclasses: `RoundStart`,
`TurnStart`, `Token`, `TurnEnd`, `VerdictRoundStart`, `Verdict`,
`TallyComplete`, `Error`. The engine owns the running `transcript` and builds
each turn's user message via `format_transcript` / `format_verdict_prompt`.
The only place the Anthropic SDK is called is `_stream_one`, which uses
`async with client.messages.stream(...)` and attaches `cache_control: ephemeral`
to each agent's system prompt so personas are cached across turns.
`load_config` returns `(agents, model, max_tokens)`.

**Renderers** — Two independent consumers of the event stream; neither calls
engine logic directly, only iterates events:

- `cli_renderer.py` — `async def render(events, console, agents)`. Emits round
  rules, colored `[ROLE]:` headers, inline streamed tokens, and the final
  tally table.
- `tui.py` — Textual `BoardRoomApp`. Layout: a `RichLog` chat column shows
  finalized turns; a `Static#current` panel pinned below it streams the
  in-flight turn token-by-token (the log can't update partial lines); a
  right sidebar contains `AgentRoster` (status per agent: pending / speaking /
  done / voted GOOD / voted BAD) and `TallySummary`. A background worker
  (`@work(exclusive=True, group="discussion")`) iterates `run_boardroom` and
  dispatches events to widget updates. Styles live in `boardroom.tcss`.

`boardroom.py` is the dispatcher: argparse, env-var validation, calls
`engine.load_config`, then runs either `_run_stdout` (which instantiates
`AsyncAnthropic` and awaits `cli_renderer.render`, via `asyncio.run`) or
`BoardRoomApp(...).run()` (the TUI's `_run_discussion` worker instantiates
its own `AsyncAnthropic`). The Anthropic client is never constructed in
the dispatcher itself.

## Conventions to preserve

- **Adding a new event type**: add the dataclass in `engine.py`, append it to
  the `Event` union, and handle it in **both** renderers (`cli_renderer.py`
  and `tui.py`'s `_run_discussion`). Forgetting one renderer is the most
  common refactor bug.
- **Verdict round modes** (`--verdict`): `decision_frame` (default) and
  `simple`. The mode is plumbed from `boardroom.py` into
  `engine.run_boardroom`'s `verdict_mode` parameter, which dispatches to
  different verdict-prompt builders and downstream rendering paths.
- **`decision_frame` mode** asks for four structured fields per agent
  (`VERDICT`, `CONFIDENCE: 1-5`, `REASONING`, `DISCONFIRMING`).
  `engine.compute_tally` turns the parsed `Verdict` list into a stratified
  tally (strong/lean/weak buckets, signed-conviction headline,
  strong-dissent flag). After `TallyComplete`, the engine emits
  `DecisionFrameStart` (renderers show a "Synthesizing…" indicator) and
  then makes one extra LLM call via `_synthesize_decision_frame` with a
  dedicated `SYNTHESIZER_SYSTEM` system prompt. The parsed result is
  yielded as a `DecisionFrame` event (case_for / case_against /
  biggest_unknown / conditions). Both renderers handle both events; the
  synthesizer's `UsageReport` flows through the cost panel like any
  other call.
- **`simple` mode** uses the original short prompt (just `VERDICT: GOOD/BAD`
  + one sentence). The engine emits a `TallyComplete` with raw `good`/`bad`/
  `overall` populated and all strata fields zero / empty `headline`. The
  renderers detect the empty headline and fall back to the legacy
  `N GOOD / M BAD → VERDICT` display. No `DecisionFrameStart` /
  `DecisionFrame` events are emitted, so no synthesis LLM call fires —
  this is the cheap / scriptable mode.
- **Streaming partial output in the TUI** goes through the `Static#current`
  widget — `RichLog.write` is for fully-formed lines only. Don't write partial
  tokens to the log directly.
- **Prompt caching** (`cache_control: ephemeral`) lives inside
  `engine._stream_one`. If you add other models / providers later, that's
  the only place the Anthropic-specific cache block needs to be branched.
- **Model IDs in use**: `claude-opus-4-7`, `claude-sonnet-4-6`,
  `claude-haiku-4-5-20251001`. Default is `claude-sonnet-4-6` in
  `engine.DEFAULT_MODEL`.
- **Textual is pinned to `0.75.1`** in `requirements.txt`. Don't bump without
  verifying the `@work(group=...)` kwarg and `RichLog` API still hold.
- **Personas** are loaded from `agents.yaml` `agents:` list (`role`, `color`,
  `system`). Any number is supported; the engine treats them as a round-robin.
- **User directives** (`RoundStart.directive`, `VerdictRoundStart.directive`)
  are fed in via the optional `consume_directive` callable passed to
  `run_boardroom`. Convention: the engine calls it once per phase boundary
  (each round start + once before the verdict round). The text is embedded
  in the user message by `format_transcript` / `format_verdict_prompt` right
  before the "It is your turn" instruction. The TUI is the only place that
  currently wires this in (via the `i` keybinding); stdout mode passes None.

## Branch & PR convention

The repo uses long-lived `claude/<topic>` feature branches. Active dev work
happens on a branch, then PRs land on `main`. When starting new work, branch
off the most recent `claude/<topic>` branch in flight (or `main` if none is).
