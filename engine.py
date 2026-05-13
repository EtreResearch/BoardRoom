"""Discussion engine for BoardRoom.

Yields a stream of events that any renderer (stdout, TUI) can consume.
"""

from __future__ import annotations

import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable

import yaml
from anthropic import APIError, AsyncAnthropic

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 600
DEFAULT_ROUNDS = 3
VERDICT_RE = re.compile(r"VERDICT:\s*(GOOD|BAD)", re.IGNORECASE)

# Anthropic per-million-token pricing in USD. Override by editing this dict
# or extending it for new model IDs.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7":          {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50},
    "claude-sonnet-4-6":        {"input":  3.0, "output": 15.0, "cache_write":  3.75, "cache_read": 0.30},
    "claude-haiku-4-5-20251001":{"input":  1.0, "output":  5.0, "cache_write":  1.25, "cache_read": 0.10},
}


@dataclass
class Agent:
    role: str
    color: str
    system: str


@dataclass
class Turn:
    speaker: str
    text: str


@dataclass
class RoundStart:
    n: int
    total: int
    order: list[str]
    directive: str | None = None


@dataclass
class TurnStart:
    role: str


@dataclass
class Token:
    role: str
    text: str


@dataclass
class TurnEnd:
    role: str
    full_text: str


@dataclass
class VerdictRoundStart:
    directive: str | None = None


@dataclass
class Verdict:
    role: str
    verdict: str  # "GOOD" | "BAD" | "UNCLEAR"
    text: str


@dataclass
class TallyComplete:
    good: int
    bad: int
    overall: str  # "GOOD" | "BAD" | "SPLIT"


@dataclass
class UsageReport:
    role: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


@dataclass
class Error:
    role: str | None
    message: str


Event = (
    RoundStart
    | TurnStart
    | Token
    | TurnEnd
    | VerdictRoundStart
    | Verdict
    | TallyComplete
    | UsageReport
    | Error
)


def compute_cost(usage: UsageReport) -> float:
    """Return estimated USD cost for a single turn. 0.0 if model is unknown."""
    rates = MODEL_PRICING.get(usage.model)
    if not rates:
        return 0.0
    return (
        usage.input_tokens                 * rates["input"]
        + usage.output_tokens              * rates["output"]
        + usage.cache_creation_input_tokens * rates["cache_write"]
        + usage.cache_read_input_tokens    * rates["cache_read"]
    ) / 1_000_000


def load_config(path: Path) -> tuple[list[Agent], str, int]:
    if not path.exists():
        sys.exit(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text())
    defaults = data.get("defaults") or {}
    model = defaults.get("model", DEFAULT_MODEL)
    max_tokens = int(defaults.get("max_tokens", DEFAULT_MAX_TOKENS))
    raw_agents = data.get("agents") or []
    if not raw_agents:
        sys.exit("agents.yaml must define at least one agent under `agents:`")
    agents = [
        Agent(role=a["role"], color=a.get("color", "white"), system=a["system"].strip())
        for a in raw_agents
    ]
    return agents, model, max_tokens


def format_transcript(
    idea: str,
    transcript: list[Turn],
    next_role: str,
    directive: str | None = None,
) -> str:
    lines = [f"IDEA: {idea}", "", "DISCUSSION SO FAR:"]
    if not transcript:
        lines.append("(you are the first to speak)")
    else:
        for turn in transcript:
            lines.append(f"[{turn.speaker}]: {turn.text}")
    if directive:
        lines += [
            "",
            f"USER DIRECTIVE: {directive}. Address this directly in your response.",
        ]
    lines += ["", f"It is your turn. Respond as the {next_role}."]
    return "\n".join(lines)


def format_verdict_prompt(
    idea: str,
    transcript: list[Turn],
    next_role: str,
    directive: str | None = None,
) -> str:
    return (
        format_transcript(idea, transcript, next_role, directive=directive)
        + "\n\nThe discussion is complete. Give your final verdict in this exact format:\n"
        "`VERDICT: GOOD` or `VERDICT: BAD`, followed by one sentence of reasoning.\n"
        "Do not write anything else."
    )


def parse_verdict(text: str) -> str:
    match = VERDICT_RE.search(text)
    return match.group(1).upper() if match else "UNCLEAR"


def overall_verdict(good: int, bad: int) -> str:
    if good > bad:
        return "GOOD"
    if bad > good:
        return "BAD"
    return "SPLIT"


async def _stream_one(
    client: AsyncAnthropic,
    agent: Agent,
    user_text: str,
    model: str,
    max_tokens: int,
) -> AsyncIterator[Event]:
    yield TurnStart(agent.role)
    chunks: list[str] = []
    final_message = None
    try:
        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": agent.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_text}],
        ) as stream:
            async for chunk in stream.text_stream:
                chunks.append(chunk)
                yield Token(agent.role, chunk)
            final_message = await stream.get_final_message()
    except APIError as e:
        yield Error(agent.role, f"API error: {e}")
        yield TurnEnd(agent.role, "".join(chunks))
        return
    yield TurnEnd(agent.role, "".join(chunks))
    if final_message is not None and getattr(final_message, "usage", None) is not None:
        u = final_message.usage
        yield UsageReport(
            role=agent.role,
            model=model,
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        )


async def run_boardroom(
    client: AsyncAnthropic,
    agents: list[Agent],
    idea: str,
    rounds: int,
    model: str,
    max_tokens: int,
    ordered: bool = False,
    seed: int | None = None,
    consume_directive: Callable[[], str | None] | None = None,
) -> AsyncIterator[Event]:
    """Drive the boardroom discussion.

    `consume_directive` is an optional zero-arg callable invoked at each
    round boundary (and before the verdict round). It should return any
    pending user directive string (and clear it from the caller's state)
    or None. The returned text is embedded in every agent's user message
    for that phase via `format_transcript` / `format_verdict_prompt`.
    """
    rng = random.Random(seed)
    transcript: list[Turn] = []

    def _next_directive() -> str | None:
        return consume_directive() if consume_directive else None

    for n in range(1, rounds + 1):
        directive = _next_directive()
        speaking_order = list(agents) if ordered else rng.sample(agents, len(agents))
        yield RoundStart(
            n=n,
            total=rounds,
            order=[a.role for a in speaking_order],
            directive=directive,
        )
        for agent in speaking_order:
            user_text = format_transcript(
                idea, transcript, agent.role, directive=directive,
            )
            full_text = ""
            async for event in _stream_one(client, agent, user_text, model, max_tokens):
                if isinstance(event, TurnEnd):
                    full_text = event.full_text
                yield event
            transcript.append(Turn(speaker=agent.role, text=full_text.strip()))

    verdict_directive = _next_directive()
    yield VerdictRoundStart(directive=verdict_directive)
    good = 0
    bad = 0
    for agent in agents:
        user_text = format_verdict_prompt(
            idea, transcript, agent.role, directive=verdict_directive,
        )
        full_text = ""
        async for event in _stream_one(client, agent, user_text, model, max_tokens):
            if isinstance(event, TurnEnd):
                full_text = event.full_text
            yield event
        verdict = parse_verdict(full_text)
        if verdict == "GOOD":
            good += 1
        elif verdict == "BAD":
            bad += 1
        yield Verdict(agent.role, verdict, full_text.strip())

    yield TallyComplete(good=good, bad=bad, overall=overall_verdict(good, bad))
