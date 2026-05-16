"""Stdout renderer that consumes engine events and prints to the terminal.

Preserves the original `boardroom.py` look: rule per round, colored
`[ROLE]:` headers, inline streamed tokens, final tally table.
"""

from __future__ import annotations

import sys
from typing import AsyncIterator

from rich.console import Console
from rich.rule import Rule
from rich.text import Text

from engine import (
    Agent,
    DecisionFrame,
    DecisionFrameStart,
    Error,
    Event,
    RoundStart,
    TallyComplete,
    Token,
    TurnEnd,
    TurnStart,
    UsageReport,
    Verdict,
    VerdictRoundStart,
    compute_cost,
)


def _color_for(role: str, agents: list[Agent]) -> str:
    for a in agents:
        if a.role == role:
            return a.color
    return "white"


def _headline_style(headline: str) -> str:
    if "GOOD" in headline:
        return "bold green"
    if "BAD" in headline:
        return "bold red"
    return "bold yellow"


async def render(
    events: AsyncIterator[Event],
    console: Console,
    agents: list[Agent],
) -> None:
    in_verdicts = False
    # Capture each agent's (verdict, confidence, reasoning) so the per-agent
    # block can show the confidence number alongside the vote.
    verdicts: dict[str, tuple[str, int | None, str | None]] = {}
    total_input = 0
    total_output = 0
    total_cache_write = 0
    total_cache_read = 0
    total_cost = 0.0

    async for event in events:
        if isinstance(event, RoundStart):
            order_text = " → ".join(event.order)
            console.print(
                Rule(
                    Text(f"Round {event.n} of {event.total}  ·  {order_text}"),
                    style="dim",
                )
            )
            if event.directive:
                console.print(Text(f"  User directive: {event.directive}", style="italic dim"))
        elif isinstance(event, VerdictRoundStart):
            in_verdicts = True
            console.print(Rule("Final verdicts", style="dim"))
            if event.directive:
                console.print(Text(f"  User directive: {event.directive}", style="italic dim"))
        elif isinstance(event, TurnStart):
            color = _color_for(event.role, agents)
            console.print(f"[bold {color}][{event.role}][/]:", end=" ")
        elif isinstance(event, Token):
            print(event.text, end="", flush=True)
        elif isinstance(event, TurnEnd):
            print("\n")
        elif isinstance(event, Verdict):
            verdicts[event.role] = (event.verdict, event.confidence, event.reasoning)
        elif isinstance(event, UsageReport):
            total_input += event.input_tokens
            total_output += event.output_tokens
            total_cache_write += event.cache_creation_input_tokens
            total_cache_read += event.cache_read_input_tokens
            total_cost += compute_cost(event)
        elif isinstance(event, TallyComplete):
            console.print(Rule("Tally", style="dim"))
            for role, (v, conf, _reason) in verdicts.items():
                marker = (
                    "[green]GOOD[/]"
                    if v == "GOOD"
                    else "[red]BAD[/]"
                    if v == "BAD"
                    else "[yellow]UNCLEAR[/]"
                )
                conf_str = f"  [dim](conf {conf}/5)[/]" if conf else ""
                console.print(f"  {role:<16} {marker}{conf_str}")

            # Stratified tally + headline + dissent flag.
            headline = event.headline or event.overall
            headline_styled = f"[{_headline_style(headline)}]{headline}[/]"
            warning = (
                f"  [yellow]⚠ {event.strong_dissent} strong dissent[/]"
                if event.strong_dissent
                else ""
            )
            console.print(f"\n  Verdict: {headline_styled}{warning}")

            # Strata line: only print sides that actually have votes.
            good_total = event.strong_good + event.lean_good + event.weak_good
            bad_total = event.strong_bad + event.lean_bad + event.weak_bad
            strata_parts: list[str] = []
            if good_total:
                strata_parts.append(
                    f"[green]{event.strong_good} strong · {event.lean_good} lean GOOD[/]"
                )
            if bad_total:
                strata_parts.append(
                    f"[red]{event.strong_bad} strong · {event.lean_bad} lean BAD[/]"
                )
            if event.weak_good or event.weak_bad:
                weak = event.weak_good + event.weak_bad
                strata_parts.append(f"[dim]{weak} weak[/]")
            if event.unclear:
                strata_parts.append(f"[yellow]{event.unclear} unclear[/]")
            if strata_parts:
                console.print("  " + "  ·  ".join(strata_parts))

            console.print(
                f"  [dim]Weighted: {event.net_score:+.0%}  "
                f"(raw: {event.good} GOOD / {event.bad} BAD)[/]"
            )
            if total_input or total_output or total_cache_read or total_cache_write:
                total_in_all = total_input + total_cache_write + total_cache_read
                cache_note = ""
                if total_cache_read or total_cache_write:
                    cache_note = (
                        f"  [dim](cache: {total_cache_write:,} w · "
                        f"{total_cache_read:,} r)[/]"
                    )
                console.print(
                    f"\n  Usage: {total_in_all:,} in · {total_output:,} out"
                    f"  [dim]~${total_cost:.4f}[/]{cache_note}"
                )
        elif isinstance(event, DecisionFrameStart):
            console.print(Rule("Decision frame", style="dim"))
            console.print("[dim italic]Synthesizing…[/]")
        elif isinstance(event, DecisionFrame):
            if event.case_for:
                console.print(Text.assemble(("Case for: ", "bold green"), event.case_for))
                console.print("")
            if event.case_against:
                console.print(Text.assemble(("Case against: ", "bold red"), event.case_against))
                console.print("")
            if event.biggest_unknown:
                console.print(Text.assemble(("Biggest unknown: ", "bold yellow"), event.biggest_unknown))
                console.print("")
            if event.conditions:
                console.print("[bold]Conditions for proceeding:[/]")
                for c in event.conditions:
                    console.print(Text.assemble(("  • ", "bold"), c))
        elif isinstance(event, Error):
            print()
            if event.role == "(synthesizer)":
                # Non-fatal: discussion + tally already displayed; just note
                # the synthesis failure and let the program finish cleanly.
                console.print(f"[red]Decision-frame error: {event.message}[/]")
            else:
                sys.exit(
                    f"Anthropic API error while {event.role or 'engine'} "
                    f"was speaking: {event.message}"
                )
