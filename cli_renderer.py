"""Stdout renderer that consumes engine events and prints to the terminal.

Preserves the original `boardroom.py` look: rule per round, colored
`[ROLE]:` headers, inline streamed tokens, final tally table.
"""

from __future__ import annotations

import sys
from typing import AsyncIterator

from rich.console import Console
from rich.rule import Rule

from engine import (
    Agent,
    Error,
    Event,
    RoundStart,
    TallyComplete,
    Token,
    TurnEnd,
    TurnStart,
    Verdict,
    VerdictRoundStart,
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

    async for event in events:
        if isinstance(event, RoundStart):
            console.print(Rule(f"Round {event.n} of {event.total}", style="dim"))
        elif isinstance(event, VerdictRoundStart):
            in_verdicts = True
            console.print(Rule("Final verdicts", style="dim"))
        elif isinstance(event, TurnStart):
            color = _color_for(event.role, agents)
            console.print(f"[bold {color}][{event.role}][/]:", end=" ")
        elif isinstance(event, Token):
            print(event.text, end="", flush=True)
        elif isinstance(event, TurnEnd):
            print("\n")
        elif isinstance(event, Verdict):
            verdicts[event.role] = event.verdict
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
        elif isinstance(event, Error):
            print()
            sys.exit(f"Anthropic API error while {event.role} was speaking: {event.message}")
