#!/usr/bin/env python3
"""BoardRoom: four virtual executives debate a business idea and deliver a verdict."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from anthropic import Anthropic, APIError
from dotenv import load_dotenv
from rich.console import Console
from rich.rule import Rule

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


def stream_turn(
    console: Console,
    client: Anthropic,
    agent: Agent,
    user_text: str,
    model: str,
    max_tokens: int,
) -> str:
    console.print(f"[bold {agent.color}][{agent.role}][/]:", end=" ")
    try:
        with client.messages.stream(
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
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
            final = stream.get_final_message()
    except APIError as e:
        print()
        sys.exit(f"Anthropic API error while {agent.role} was speaking: {e}")
    print("\n")
    return "".join(b.text for b in final.content if getattr(b, "text", None))


def run_discussion(
    console: Console,
    client: Anthropic,
    agents: list[Agent],
    idea: str,
    rounds: int,
    model: str,
    max_tokens: int,
) -> list[Turn]:
    transcript: list[Turn] = []
    for n in range(1, rounds + 1):
        console.print(Rule(f"Round {n} of {rounds}", style="dim"))
        for agent in agents:
            user_text = format_transcript(idea, transcript, agent.role)
            text = stream_turn(console, client, agent, user_text, model, max_tokens)
            transcript.append(Turn(speaker=agent.role, text=text.strip()))
    return transcript


def run_verdict_round(
    console: Console,
    client: Anthropic,
    agents: list[Agent],
    idea: str,
    transcript: list[Turn],
    model: str,
    max_tokens: int,
) -> dict[str, str]:
    console.print(Rule("Final verdicts", style="dim"))
    verdicts: dict[str, str] = {}
    for agent in agents:
        user_text = format_verdict_prompt(idea, transcript, agent.role)
        text = stream_turn(console, client, agent, user_text, model, max_tokens)
        match = VERDICT_RE.search(text)
        verdicts[agent.role] = match.group(1).upper() if match else "UNCLEAR"
    return verdicts


def summarize(console: Console, verdicts: dict[str, str]) -> None:
    console.print(Rule("Tally", style="dim"))
    good = sum(1 for v in verdicts.values() if v == "GOOD")
    bad = sum(1 for v in verdicts.values() if v == "BAD")
    for role, v in verdicts.items():
        marker = "[green]GOOD[/]" if v == "GOOD" else "[red]BAD[/]" if v == "BAD" else "[yellow]UNCLEAR[/]"
        console.print(f"  {role:<16} {marker}")
    if good > bad:
        overall = "[bold green]GOOD[/]"
    elif bad > good:
        overall = "[bold red]BAD[/]"
    else:
        overall = "[bold yellow]SPLIT[/]"
    console.print(f"\n  Tally: {good} GOOD / {bad} BAD  →  Verdict: {overall}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Four virtual executives debate a business idea and deliver a verdict.",
    )
    parser.add_argument("idea", help="The business idea to evaluate (quote it).")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help=f"Discussion rounds before verdict (default {DEFAULT_ROUNDS}).")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "agents.yaml", help="Path to agents.yaml.")
    parser.add_argument("--model", default=None, help="Override the model ID for all agents.")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to .env or export it.")

    agents, model, max_tokens = load_config(args.config)
    if args.model:
        model = args.model

    console = Console()
    console.print(Rule("BoardRoom", style="bold"))
    console.print(f"[bold]Idea:[/] {args.idea}")
    console.print(f"[dim]Model: {model} · Rounds: {args.rounds} · Agents: {', '.join(a.role for a in agents)}[/]\n")

    client = Anthropic()
    transcript = run_discussion(console, client, agents, args.idea, args.rounds, model, max_tokens)
    verdicts = run_verdict_round(console, client, agents, args.idea, transcript, model, max_tokens)
    summarize(console, verdicts)


if __name__ == "__main__":
    main()
