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
REASONING_RE = re.compile(
    r"REASONING:\s*(.+?)(?=DISCONFIRMING:|\Z)", re.IGNORECASE | re.DOTALL
)
DISCONFIRMING_RE = re.compile(
    r"DISCONFIRMING:\s*(.+?)\Z", re.IGNORECASE | re.DOTALL
)

# Decision-frame synthesis output parsers (one field stops at the next).
CASE_FOR_RE = re.compile(
    r"CASE_FOR:\s*(.+?)(?=CASE_AGAINST:|BIGGEST_UNKNOWN:|CONDITIONS:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
CASE_AGAINST_RE = re.compile(
    r"CASE_AGAINST:\s*(.+?)(?=BIGGEST_UNKNOWN:|CONDITIONS:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
BIGGEST_UNKNOWN_RE = re.compile(
    r"BIGGEST_UNKNOWN:\s*(.+?)(?=CONDITIONS:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
CONDITIONS_RE = re.compile(
    r"CONDITIONS:\s*(.+?)\Z", re.IGNORECASE | re.DOTALL
)

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
    disconfirming: str | None = None  # parsed DISCONFIRMING line, None if unparseable


@dataclass
class DecisionFrameStart:
    """Marker that the engine is about to synthesize the decision frame."""
    pass


@dataclass
class DecisionFrame:
    """Synthesized neutral summary of the panel's verdicts."""
    case_for: str
    case_against: str
    biggest_unknown: str
    conditions: list[str]


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
class ScoreReport:
    """One agent's per-dimension scorecard (verdict_mode='scorecard')."""
    role: str
    scores: dict[str, int | None]   # {"MARKET": 4, "TECH": None, ...}
    notes: dict[str, str]            # {"MARKET": "the one-sentence note", ...}
    text: str                        # raw response


@dataclass
class ScorecardComplete:
    """Aggregated scorecard across all agents."""
    by_agent: list[ScoreReport]
    averages: dict[str, float]   # per-dimension mean across agents with that score
    composite: float              # mean of the four dimension averages


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
    | DecisionFrameStart
    | DecisionFrame
    | ScoreReport
    | ScorecardComplete
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
        "DISCONFIRMING: The single strongest reason your vote might be wrong, in one sentence.\n"
        "\n"
        "Rules:\n"
        "- VERDICT must be exactly GOOD or BAD.\n"
        "- CONFIDENCE is an integer 1-5 (1 = borderline / barely decided, 3 = leaning,"
        " 5 = strongly held). Be honest: pick 1-2 if you're on the fence, 4-5 only if"
        " the case is compelling.\n"
        "- REASONING is one or two sentences.\n"
        "- DISCONFIRMING: even though you voted, articulate the most credible counter-"
        "argument against your own position. Push yourself out of confirmation bias."
        " One sentence.\n"
        "- Output nothing else — no preamble, no markdown."
    )


def format_scorecard_prompt(
    idea: str,
    transcript: list[Turn],
    next_role: str,
    directive: str | None = None,
) -> str:
    """Scorecard verdict prompt for `--verdict scorecard`.

    Asks for 1-5 ratings across four dimensions with one-sentence
    justifications. RISK is framed "higher = safer" so all dimensions point
    the same direction and per-dimension averaging is coherent.
    """
    return (
        format_transcript(idea, transcript, next_role, directive=directive)
        + "\n\nThe discussion is complete. Score this idea on four dimensions"
        " using EXACTLY this format:\n"
        "\n"
        "MARKET: 4 - One sentence on market opportunity / timing (higher = bigger market or better timing).\n"
        "TECH: 3 - One sentence on technical feasibility (higher = more feasible / less build risk).\n"
        "UX: 4 - One sentence on user value clarity (higher = clearer value to the user).\n"
        "RISK: 2 - One sentence on risk profile (higher = better controlled / safer).\n"
        "\n"
        "Rules:\n"
        "- Each score is an integer 1-5.\n"
        "- Each line is exactly: <DIMENSION>: <score> - <one sentence>\n"
        "- Use the dimension keywords MARKET, TECH, UX, RISK in that order.\n"
        "- Output exactly four lines. No preamble, no closing remarks."
    )


def parse_scorecard(text: str) -> tuple[dict[str, int | None], dict[str, str]]:
    """Parse a scorecard response. Returns (scores, notes), keyed by dimension.

    Missing dimensions get `None` (scores) or `""` (notes). Unparseable
    output → all dimensions empty/None; the caller treats that as a
    degraded-but-honest result.
    """
    scores: dict[str, int | None] = {d: None for d in SCORECARD_DIMENSIONS}
    notes: dict[str, str] = {d: "" for d in SCORECARD_DIMENSIONS}
    for m in SCORECARD_LINE_RE.finditer(text):
        dim = m.group(1).upper()
        scores[dim] = int(m.group(2))
        notes[dim] = m.group(3).strip()
    return scores, notes


def compute_scorecard_complete(reports: list["ScoreReport"]) -> "ScorecardComplete":
    """Average each dimension across agents (skipping None) and the composite."""
    averages: dict[str, float] = {}
    for dim in SCORECARD_DIMENSIONS:
        values = [r.scores[dim] for r in reports if r.scores.get(dim) is not None]
        averages[dim] = round(sum(values) / len(values), 2) if values else 0.0
    nonzero = [v for v in averages.values() if v > 0]
    composite = round(sum(nonzero) / len(nonzero), 2) if nonzero else 0.0
    return ScorecardComplete(by_agent=list(reports), averages=averages, composite=composite)


def format_simple_verdict_prompt(
    idea: str,
    transcript: list[Turn],
    next_role: str,
    directive: str | None = None,
) -> str:
    """Compact verdict prompt for `--verdict simple`: just GOOD/BAD + one sentence.

    No confidence, no disconfirming, no synthesis — minimum tokens, minimum
    cost, one-line tally suitable for batch / scripted comparison.
    """
    return (
        format_transcript(idea, transcript, next_role, directive=directive)
        + "\n\nThe discussion is complete. Give your final verdict in this exact format:\n"
        "`VERDICT: GOOD` or `VERDICT: BAD`, followed by one sentence of reasoning.\n"
        "Do not write anything else."
    )


SYNTHESIZER_SYSTEM = (
    "You are the panel's secretary. Your job is to synthesize a neutral "
    "decision frame from the executives' verdicts and the discussion that "
    "led to them. You do not have an opinion; you reflect the panel's "
    "thinking honestly, including dissent — especially when it's a minority."
)


def format_decision_frame_prompt(
    idea: str,
    transcript: list[Turn],
    verdicts: list[Verdict],
    tally_headline: str,
    strong_dissent: int,
) -> str:
    """Build the user message for the decision-frame synthesis call."""
    lines = [f"IDEA: {idea}", ""]
    if transcript:
        lines.append("DISCUSSION:")
        for turn in transcript:
            lines.append(f"[{turn.speaker}]: {turn.text}")
        lines.append("")
    lines.append("PER-AGENT VERDICTS:")
    for v in verdicts:
        conf = f"conf {v.confidence}/5" if v.confidence else "conf —"
        reason = v.reasoning or "(no reasoning)"
        disconfirm = v.disconfirming or "(none stated)"
        lines.append(
            f"- [{v.role}] {v.verdict} ({conf}). Reasoning: {reason} "
            f"Their own counter-argument: {disconfirm}"
        )
    lines.append("")
    lines.append(f"TALLY HEADLINE: {tally_headline}")
    if strong_dissent:
        lines.append(
            f"STRONG DISSENT: {strong_dissent} agent(s) voted opposite the majority "
            "with high confidence. Capture their objection prominently."
        )
    lines += [
        "",
        "Synthesize a decision frame in EXACTLY this format:",
        "",
        "CASE_FOR: Two sentences. The strongest affirmative argument, synthesized across "
        "the panel — not a copy of any one agent's words.",
        "CASE_AGAINST: Two sentences. The strongest objection, even when it's a "
        "minority view. This is where strong dissent goes.",
        "BIGGEST_UNKNOWN: One sentence. The single question whose answer would most "
        "change the panel's view. A specific unknown, not a generic risk.",
        "CONDITIONS:",
        "- Three to five bullets. Concrete preconditions that must hold for the idea to work.",
        "- Each bullet is a single line, terse and actionable.",
        "",
        "Rules:",
        "- Be neutral. Don't editorialize.",
        "- Don't reuse the agents' phrasing verbatim — synthesize.",
        "- Don't soften minority dissent.",
        "- Output only the four sections. No preamble, no closing remarks.",
    ]
    return "\n".join(lines)


async def _synthesize_decision_frame(
    client: AsyncAnthropic,
    idea: str,
    transcript: list[Turn],
    verdicts: list[Verdict],
    tally: TallyComplete,
    model: str,
) -> AsyncIterator[Event]:
    """One API call to synthesize a `DecisionFrame`.

    Yields a `UsageReport` (so cost tracking stays accurate) followed by
    a `DecisionFrame`, or an `Error` if the call fails. Non-streaming —
    the synthesized output is short and arrives as a single block.
    """
    user_text = format_decision_frame_prompt(
        idea, transcript, verdicts,
        tally_headline=tally.headline or tally.overall,
        strong_dissent=tally.strong_dissent,
    )
    chunks: list[str] = []
    final_message = None
    try:
        async with client.messages.stream(
            model=model,
            max_tokens=500,  # Synthesis output is short and bounded.
            system=SYNTHESIZER_SYSTEM,
            messages=[{"role": "user", "content": user_text}],
        ) as stream:
            async for chunk in stream.text_stream:
                chunks.append(chunk)
            final_message = await stream.get_final_message()
    except APIError as e:
        yield Error("(synthesizer)", f"Decision-frame synthesis failed: {e}")
        return

    if final_message is not None and getattr(final_message, "usage", None) is not None:
        u = final_message.usage
        yield UsageReport(
            role="(synthesizer)",
            model=model,
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        )

    yield parse_decision_frame("".join(chunks))


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


def parse_disconfirming(text: str) -> str | None:
    """Parse `DISCONFIRMING: ...` block. Returns None if missing."""
    match = DISCONFIRMING_RE.search(text)
    if not match:
        return None
    disconfirming = match.group(1).strip()
    return disconfirming or None


def parse_decision_frame(text: str) -> DecisionFrame:
    """Parse a synthesis response into a `DecisionFrame`.

    Missing or empty sections degrade to empty strings / lists rather than
    raising — the renderer can choose how to handle partial output.
    """
    def _grab(regex: re.Pattern) -> str:
        m = regex.search(text)
        return m.group(1).strip() if m else ""

    case_for = _grab(CASE_FOR_RE)
    case_against = _grab(CASE_AGAINST_RE)
    biggest_unknown = _grab(BIGGEST_UNKNOWN_RE)

    conditions: list[str] = []
    m = CONDITIONS_RE.search(text)
    if m:
        block = m.group(1).strip()
        for line in block.splitlines():
            stripped = line.strip()
            # Accept "- ...", "* ...", or "• ..." bullets.
            if stripped.startswith(("-", "*", "•")):
                content = stripped[1:].strip()
                if content:
                    conditions.append(content)

    return DecisionFrame(
        case_for=case_for,
        case_against=case_against,
        biggest_unknown=biggest_unknown,
        conditions=conditions,
    )


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


VERDICT_MODES = ("decision_frame", "simple", "scorecard")
DEFAULT_VERDICT_MODE = "decision_frame"

SCORECARD_DIMENSIONS = ("MARKET", "TECH", "UX", "RISK")
SCORECARD_LINE_RE = re.compile(
    r"(MARKET|TECH|UX|RISK):\s*([1-5])\s*[-—:]\s*(.+?)(?=\n\s*[A-Z]+:|\Z)",
    re.IGNORECASE | re.DOTALL,
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
    verdict_mode: str = DEFAULT_VERDICT_MODE,
) -> AsyncIterator[Event]:
    """Drive the boardroom discussion.

    `consume_directive` is an optional zero-arg callable invoked at each
    round boundary (and before the verdict round). It should return any
    pending user directive string (and clear it from the caller's state)
    or None. The returned text is embedded in every agent's user message
    for that phase via `format_transcript` / `format_verdict_prompt`.

    `verdict_mode` controls how the verdict round is structured:
    - `"decision_frame"` (default): each agent gives VERDICT + CONFIDENCE +
      REASONING + DISCONFIRMING; the engine renders a stratified tally and
      makes one extra LLM call to synthesize a decision frame.
    - `"simple"`: each agent gives VERDICT + one sentence of reasoning.
      No confidence, no synthesis call. Cheaper; suited for batch /
      scripted runs that just want one number per idea.
    """
    if verdict_mode not in VERDICT_MODES:
        raise ValueError(
            f"Unknown verdict_mode: {verdict_mode!r}. "
            f"Expected one of: {', '.join(VERDICT_MODES)}"
        )

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

    # Scorecard mode has a different per-agent shape (no GOOD/BAD vote) so
    # it gets its own path. The decision_frame and simple modes both produce
    # Verdict events and share the per-agent loop below.
    if verdict_mode == "scorecard":
        score_reports: list[ScoreReport] = []
        for agent in agents:
            user_text = format_scorecard_prompt(
                idea, transcript, agent.role, directive=verdict_directive,
            )
            full_text = ""
            async for event in _stream_one(client, agent, user_text, model, max_tokens):
                if isinstance(event, TurnEnd):
                    full_text = event.full_text
                yield event
            scores, notes = parse_scorecard(full_text)
            sr = ScoreReport(
                role=agent.role,
                scores=scores,
                notes=notes,
                text=full_text.strip(),
            )
            score_reports.append(sr)
            yield sr
        yield compute_scorecard_complete(score_reports)
        return

    collected_verdicts: list[Verdict] = []

    # The verdict prompt and the per-agent Verdict shape both depend on the mode.
    use_simple = verdict_mode == "simple"
    prompt_builder = (
        format_simple_verdict_prompt if use_simple else format_verdict_prompt
    )

    for agent in agents:
        user_text = prompt_builder(
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
            # Simple mode doesn't ask for these and shouldn't pretend to
            # have them. Leaving them None keeps `compute_tally`'s output
            # in a degraded-but-honest "no strata" state.
            confidence=None if use_simple else parse_confidence(full_text),
            reasoning=None if use_simple else parse_reasoning(full_text),
            disconfirming=None if use_simple else parse_disconfirming(full_text),
        )
        collected_verdicts.append(v)
        yield v

    if use_simple:
        # Raw tally only — no strata math, no headline, no synthesis.
        good = bad = unclear = 0
        for v in collected_verdicts:
            if v.verdict == "GOOD":
                good += 1
            elif v.verdict == "BAD":
                bad += 1
            else:
                unclear += 1
        yield TallyComplete(
            good=good,
            bad=bad,
            overall=overall_verdict(good, bad),
            unclear=unclear,
            # All other strata fields default to 0 / "" — renderers detect this
            # via empty `headline` and fall back to the legacy display.
        )
        return

    tally = TallyComplete(**compute_tally(collected_verdicts))
    yield tally

    # Synthesize the decision frame from the verdicts + discussion.
    # Adds one extra LLM call per run, bounded to a small max_tokens.
    yield DecisionFrameStart()
    async for event in _synthesize_decision_frame(
        client, idea, transcript, collected_verdicts, tally, model,
    ):
        yield event
