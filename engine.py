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
CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*([1-5])", re.IGNORECASE)
REASONING_RE = re.compile(r"REASONING:\s*(.+?)(?:\n\s*$|\Z)", re.IGNORECASE | re.DOTALL)

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
    confidence: int | None = None  # 1-5, None if unparseable
    reasoning: str | None = None   # parsed REASONING line, None if unparseable


@dataclass
class TallyComplete:
    good: int                       # raw count of GOOD votes (legacy)
    bad: int                        # raw count of BAD votes (legacy)
    overall: str                    # "GOOD" | "BAD" | "SPLIT" (raw-count based)
    # Confidence-aware stratification. Defaults preserve a degenerate tally when
    # no confidence data is parseable.
    strong_good: int = 0
    lean_good: int = 0
    weak_good: int = 0
    strong_bad: int = 0
    lean_bad: int = 0
    weak_bad: int = 0
    unclear: int = 0
    net_score: float = 0.0          # signed conviction in [-1.0, +1.0]
    headline: str = ""              # e.g. "moderate GOOD", "weak BAD", "deeply split"
    strong_dissent: int = 0          # agents who voted opposite the majority with confidence ≥ 4


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
        "\n"
        "VERDICT: GOOD\n"
        "CONFIDENCE: 4\n"
        "REASONING: One or two sentences explaining your vote.\n"
        "\n"
        "Rules:\n"
        "- VERDICT must be exactly GOOD or BAD.\n"
        "- CONFIDENCE is an integer 1-5 (1 = borderline / barely decided, 3 = leaning,"
        " 5 = strongly held). Be honest: pick 1-2 if you're on the fence, 4-5 only if"
        " the case is compelling.\n"
        "- REASONING is one or two sentences.\n"
        "- Output nothing else — no preamble, no markdown."
    )


def parse_verdict(text: str) -> str:
    match = VERDICT_RE.search(text)
    return match.group(1).upper() if match else "UNCLEAR"


def parse_confidence(text: str) -> int | None:
    """Parse `CONFIDENCE: N` (1-5). Returns None if missing or out of range."""
    match = CONFIDENCE_RE.search(text)
    return int(match.group(1)) if match else None


def parse_reasoning(text: str) -> str | None:
    """Parse `REASONING: ...` block. Returns None if missing."""
    match = REASONING_RE.search(text)
    if not match:
        return None
    reasoning = match.group(1).strip()
    return reasoning or None


def overall_verdict(good: int, bad: int) -> str:
    if good > bad:
        return "GOOD"
    if bad > good:
        return "BAD"
    return "SPLIT"


def _bucket(confidence: int | None) -> str:
    """Return 'strong' | 'lean' | 'weak' based on confidence (treating None as weak)."""
    if confidence is None:
        return "weak"
    if confidence >= 4:
        return "strong"
    if confidence >= 2:
        return "lean"
    return "weak"


def compute_tally(verdicts: list["Verdict"]) -> dict:
    """Compute the stratified tally + headline + dissent from a list of Verdicts.

    Returns a dict matching the new fields on `TallyComplete`.
    """
    strong_good = lean_good = weak_good = 0
    strong_bad = lean_bad = weak_bad = 0
    unclear = 0
    signed_total = 0.0

    for v in verdicts:
        if v.verdict == "GOOD":
            bucket = _bucket(v.confidence)
            if bucket == "strong":
                strong_good += 1
            elif bucket == "lean":
                lean_good += 1
            else:
                weak_good += 1
            signed_total += (v.confidence if v.confidence else 1)
        elif v.verdict == "BAD":
            bucket = _bucket(v.confidence)
            if bucket == "strong":
                strong_bad += 1
            elif bucket == "lean":
                lean_bad += 1
            else:
                weak_bad += 1
            signed_total -= (v.confidence if v.confidence else 1)
        else:
            unclear += 1

    good_count = strong_good + lean_good + weak_good
    bad_count = strong_bad + lean_bad + weak_bad

    # Max possible signed conviction = ±5 per agent in the round. UNCLEAR
    # agents are included so they dilute the conviction (1 strong GOOD + 3
    # UNCLEAR should not read as "strong GOOD" — the panel is uncertain).
    max_total = len(verdicts) * 5 if verdicts else 1
    net_score = signed_total / max_total

    # Headline: direction + conviction strength.
    if good_count == 0 and bad_count == 0:
        headline = "no clear verdict"
    elif good_count == bad_count:
        # Tie by count. Strength of the tie matters: high-confidence on both
        # sides is *worse* than a wishy-washy tie because it means real
        # disagreement.
        if (strong_good + strong_bad) >= 1:
            headline = "deeply split"
        else:
            headline = "split"
    elif good_count > bad_count:
        abs_net = abs(net_score)
        if abs_net >= 0.60:
            headline = "strong GOOD"
        elif abs_net >= 0.25:
            headline = "moderate GOOD"
        else:
            headline = "weak GOOD"
    else:
        abs_net = abs(net_score)
        if abs_net >= 0.60:
            headline = "strong BAD"
        elif abs_net >= 0.25:
            headline = "moderate BAD"
        else:
            headline = "weak BAD"

    # Strong dissent = agents who voted opposite the majority with confidence ≥ 4.
    if good_count > bad_count:
        strong_dissent = strong_bad
    elif bad_count > good_count:
        strong_dissent = strong_good
    else:
        # Tie: both sides count as dissent.
        strong_dissent = strong_good + strong_bad

    return {
        "good": good_count,
        "bad": bad_count,
        "overall": overall_verdict(good_count, bad_count),
        "strong_good": strong_good,
        "lean_good": lean_good,
        "weak_good": weak_good,
        "strong_bad": strong_bad,
        "lean_bad": lean_bad,
        "weak_bad": weak_bad,
        "unclear": unclear,
        "net_score": net_score,
        "headline": headline,
        "strong_dissent": strong_dissent,
    }


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
    collected_verdicts: list[Verdict] = []
    for agent in agents:
        user_text = format_verdict_prompt(
            idea, transcript, agent.role, directive=verdict_directive,
        )
        full_text = ""
        async for event in _stream_one(client, agent, user_text, model, max_tokens):
            if isinstance(event, TurnEnd):
                full_text = event.full_text
            yield event
        v = Verdict(
            role=agent.role,
            verdict=parse_verdict(full_text),
            text=full_text.strip(),
            confidence=parse_confidence(full_text),
            reasoning=parse_reasoning(full_text),
        )
        collected_verdicts.append(v)
        yield v

    yield TallyComplete(**compute_tally(collected_verdicts))
