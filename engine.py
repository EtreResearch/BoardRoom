"""Discussion engine for BoardRoom.

Yields a stream of events that any renderer (stdout, TUI) can consume.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import yaml

from providers import Provider

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 600
DEFAULT_ROUNDS = 3
VERDICT_RE = re.compile(r"VERDICT:\s*(GOOD|BAD)", re.IGNORECASE)


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
    pass


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
    | Error
)


@dataclass
class Defaults:
    provider: str
    model: str
    max_tokens: int
    base_url: str | None
    api_key_env: str | None


def load_config(path: Path) -> tuple[list[Agent], Defaults]:
    if not path.exists():
        sys.exit(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text())
    raw_defaults = data.get("defaults") or {}
    defaults = Defaults(
        provider=raw_defaults.get("provider", DEFAULT_PROVIDER),
        model=raw_defaults.get("model", DEFAULT_MODEL),
        max_tokens=int(raw_defaults.get("max_tokens", DEFAULT_MAX_TOKENS)),
        base_url=raw_defaults.get("base_url"),
        api_key_env=raw_defaults.get("api_key_env"),
    )
    raw_agents = data.get("agents") or []
    if not raw_agents:
        sys.exit("agents.yaml must define at least one agent under `agents:`")
    agents = [
        Agent(role=a["role"], color=a.get("color", "white"), system=a["system"].strip())
        for a in raw_agents
    ]
    return agents, defaults


def format_transcript(idea: str, transcript: list[Turn], next_role: str) -> str:
    lines = [f"IDEA: {idea}", "", "DISCUSSION SO FAR:"]
    if not transcript:
        lines.append("(you are the first to speak)")
    else:
        for turn in transcript:
            lines.append(f"[{turn.speaker}]: {turn.text}")
    lines += ["", f"It is your turn. Respond as the {next_role}."]
    return "\n".join(lines)


def format_verdict_prompt(idea: str, transcript: list[Turn], next_role: str) -> str:
    return (
        format_transcript(idea, transcript, next_role)
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
    provider: Provider,
    agent: Agent,
    user_text: str,
    model: str,
    max_tokens: int,
) -> AsyncIterator[Event]:
    yield TurnStart(agent.role)
    chunks: list[str] = []
    try:
        async for chunk in provider.stream(
            system=agent.system,
            user_message=user_text,
            model=model,
            max_tokens=max_tokens,
        ):
            chunks.append(chunk)
            yield Token(agent.role, chunk)
    except Exception as e:
        yield Error(agent.role, f"{type(e).__name__}: {e}")
        yield TurnEnd(agent.role, "".join(chunks))
        return
    yield TurnEnd(agent.role, "".join(chunks))


async def run_boardroom(
    provider: Provider,
    agents: list[Agent],
    idea: str,
    rounds: int,
    model: str,
    max_tokens: int,
) -> AsyncIterator[Event]:
    transcript: list[Turn] = []

    for n in range(1, rounds + 1):
        yield RoundStart(n, rounds)
        for agent in agents:
            user_text = format_transcript(idea, transcript, agent.role)
            full_text = ""
            async for event in _stream_one(provider, agent, user_text, model, max_tokens):
                if isinstance(event, TurnEnd):
                    full_text = event.full_text
                yield event
            transcript.append(Turn(speaker=agent.role, text=full_text.strip()))

    yield VerdictRoundStart()
    good = 0
    bad = 0
    for agent in agents:
        user_text = format_verdict_prompt(idea, transcript, agent.role)
        full_text = ""
        async for event in _stream_one(provider, agent, user_text, model, max_tokens):
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
