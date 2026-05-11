#!/usr/bin/env python3
"""BoardRoom: four virtual executives debate a business idea and deliver a verdict."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.rule import Rule

from engine import DEFAULT_ROUNDS, Defaults, load_config, run_boardroom
from providers import PROVIDER_PRESETS, Provider, make_provider

DEFAULT_CONFIG = Path(__file__).parent / "agents.yaml"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Four virtual executives debate a business idea and deliver a verdict.",
    )
    parser.add_argument("idea", help="The business idea to evaluate (quote it).")
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"Discussion rounds before verdict (default {DEFAULT_ROUNDS}).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to agents.yaml.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDER_PRESETS),
        default=None,
        help="Override the LLM provider for this run.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model ID for all agents.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the provider base URL (for openai-compatible providers).",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Name of the env var that holds the provider API key.",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch the full-screen Textual UI instead of streaming to stdout.",
    )
    return parser.parse_args(argv)


async def _run_stdout(
    args: argparse.Namespace,
    provider: Provider,
    agents,
    model: str,
    max_tokens: int,
) -> None:
    from cli_renderer import render

    console = Console()
    console.print(Rule("BoardRoom", style="bold"))
    console.print(f"[bold]Idea:[/] {args.idea}")
    console.print(
        f"[dim]Provider: {provider.name} · Model: {model} · Rounds: {args.rounds} · "
        f"Agents: {', '.join(a.role for a in agents)}[/]\n"
    )

    try:
        events = run_boardroom(provider, agents, args.idea, args.rounds, model, max_tokens)
        await render(events, console, agents)
    finally:
        await provider.close()


def _run_tui(
    args: argparse.Namespace,
    provider: Provider,
    agents,
    model: str,
    max_tokens: int,
) -> None:
    from tui import BoardRoomApp

    app = BoardRoomApp(
        idea=args.idea,
        agents=agents,
        rounds=args.rounds,
        model=model,
        max_tokens=max_tokens,
        provider=provider,
    )
    app.run()


def _resolve_provider(args: argparse.Namespace, defaults: Defaults) -> Provider:
    provider_key = args.provider or defaults.provider
    base_url = args.base_url or defaults.base_url
    api_key_env = args.api_key_env or defaults.api_key_env
    api_key = os.getenv(api_key_env) if api_key_env else None
    try:
        return make_provider(provider_key, base_url=base_url, api_key=api_key)
    except RuntimeError as e:
        sys.exit(str(e))
    except ValueError as e:
        sys.exit(str(e))


def _run_init() -> int:
    from init import run_init
    return run_init(DEFAULT_CONFIG)


def main() -> None:
    load_dotenv()

    if len(sys.argv) >= 2 and sys.argv[1] == "init":
        sys.exit(_run_init())

    args = _parse_args(sys.argv[1:])
    agents, defaults = load_config(args.config)
    model = args.model or defaults.model
    provider = _resolve_provider(args, defaults)

    if args.tui:
        _run_tui(args, provider, agents, model, defaults.max_tokens)
    else:
        asyncio.run(_run_stdout(args, provider, agents, model, defaults.max_tokens))


if __name__ == "__main__":
    main()
