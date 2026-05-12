#!/usr/bin/env python3
"""BoardRoom: four virtual executives debate a business idea and deliver a verdict."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.rule import Rule

from engine import DEFAULT_ROUNDS, load_config, run_boardroom

HIGH_ROUNDS_WARN_THRESHOLD = 10


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}")
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"must be at least 1 (got {n}); zero or negative rounds would skip the discussion entirely"
        )
    return n


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Four virtual executives debate a business idea and deliver a verdict.",
    )
    parser.add_argument("idea", help="The business idea to evaluate (quote it).")
    parser.add_argument(
        "--rounds",
        type=_positive_int,
        default=DEFAULT_ROUNDS,
        help=f"Discussion rounds before verdict (default {DEFAULT_ROUNDS}, minimum 1).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "agents.yaml",
        help="Path to agents.yaml.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model ID for all agents.",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch the full-screen Textual UI instead of streaming to stdout.",
    )
    parser.add_argument(
        "--ordered",
        action="store_true",
        help="Speak in YAML order every round (default: shuffle each round).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the per-round shuffle for reproducibility.",
    )
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Skip the TUI setup screen and use the flag values directly.",
    )
    return parser.parse_args()


async def _run_stdout(args, agents, model, max_tokens) -> None:
    from cli_renderer import render

    console = Console()
    console.print(Rule("BoardRoom", style="bold"))
    console.print(f"[bold]Idea:[/] {args.idea}")
    console.print(
        f"[dim]Model: {model} · Rounds: {args.rounds} · "
        f"Agents: {', '.join(a.role for a in agents)}[/]\n"
    )

    client = AsyncAnthropic()
    try:
        events = run_boardroom(
            client,
            agents,
            args.idea,
            args.rounds,
            model,
            max_tokens,
            ordered=args.ordered,
            seed=args.seed,
        )
        await render(events, console, agents)
    finally:
        await client.close()


def _run_tui(args, agents, model, max_tokens) -> None:
    from tui import BoardRoomApp

    app = BoardRoomApp(
        idea=args.idea,
        agents=agents,
        rounds=args.rounds,
        model=model,
        max_tokens=max_tokens,
        ordered=args.ordered,
        seed=args.seed,
        no_setup=args.no_setup,
    )
    app.run()


def main() -> None:
    args = _parse_args()

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to .env or export it.")

    agents, model, max_tokens = load_config(args.config)
    if args.model:
        model = args.model

    if args.rounds > HIGH_ROUNDS_WARN_THRESHOLD:
        total_calls = (args.rounds + 1) * len(agents)
        print(
            f"Warning: --rounds {args.rounds} × {len(agents)} agents = "
            f"~{total_calls} API calls (incl. verdict round). "
            f"This may be expensive on paid providers. Ctrl+C to cancel.",
            file=sys.stderr,
        )

    if args.tui:
        _run_tui(args, agents, model, max_tokens)
    else:
        asyncio.run(_run_stdout(args, agents, model, max_tokens))


if __name__ == "__main__":
    main()
