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


async def render(
    events: AsyncIterator[Event],
    console: Console,
    agents: list[Agent],
) -> None:
    in_verdicts = False
    verdicts: dict[str, str] = {}
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
            verdicts[event.role] = event.verdict
        elif isinstance(event, UsageReport):
            total_input += event.input_tokens
            total_output += event.output_tokens
            total_cache_write += event.cache_creation_input_tokens
            total_cache_read += event.cache_read_input_tokens
            total_cost += compute_cost(event)
        elif isinstance(event, TallyComplete):
            console.print(Rule("Tally", style="dim"))
            for role, v in verdicts.items():
                marker = (
                    "[green]GOOD[/]"
                    if v == "GOOD"
                    else "[red]BAD[/]"
                    if v == "BAD"
                    else "[yellow]UNCLEAR[/]"
                )
                console.print(f"  {role:<16} {marker}")
            overall_styled = {
                "GOOD": "[bold green]GOOD[/]",
                "BAD": "[bold red]BAD[/]",
                "SPLIT": "[bold yellow]SPLIT[/]",
            }[event.overall]
            console.print(
                f"\n  Tally: {event.good} GOOD / {event.bad} BAD  →  Verdict: {overall_styled}"
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
        elif isinstance(event, Error):
            print()
            sys.exit(f"Anthropic API error while {event.role} was speaking: {event.message}")
