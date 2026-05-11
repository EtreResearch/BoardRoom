"""First-run setup wizard.

Walks the user through picking a provider and a model, then writes the
choice into `agents.yaml` under `defaults:`. Local providers (Ollama,
LM Studio, ...) are listed first since they're the affordability story.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from providers import PROVIDER_PRESETS, make_provider


# Display order: local first, cloud after.
PROVIDER_ORDER = [
    "ollama",
    "lmstudio",
    "vllm",
    "localai",
    "anthropic",
    "openai",
    "openrouter",
    "groq",
    "together",
    "custom",
]


def run_init(config_path: Path) -> int:
    console = Console()
    console.print(
        Panel.fit(
            "[bold]BoardRoom setup[/]\n\n"
            "Pick the LLM provider you want the four executives to use.\n"
            "Local options like [bold]Ollama[/] cost nothing.",
            border_style="cyan",
        )
    )

    provider_key = _pick_provider(console)
    preset = PROVIDER_PRESETS[provider_key]
    base_url = _resolve_base_url(console, provider_key, preset)
    api_key_env = _resolve_api_key_env(console, provider_key, preset)
    model = asyncio.run(_pick_model(console, provider_key, base_url, preset, api_key_env))

    _write_config(
        console,
        config_path,
        provider=provider_key,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
    )

    console.print()
    console.print(
        Panel.fit(
            "[bold green]Setup complete.[/]\n\n"
            "Try it:\n"
            "  [cyan]python boardroom.py \"An app that uses AI to plan toddler meals.\"[/]\n"
            "Or with the TUI:\n"
            "  [cyan]python boardroom.py --tui \"...\"[/]",
            border_style="green",
        )
    )
    return 0


def _pick_provider(console: Console) -> str:
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Key")
    table.add_column("Description")
    for i, key in enumerate(PROVIDER_ORDER, 1):
        table.add_row(str(i), f"[cyan]{key}[/]", PROVIDER_PRESETS[key]["label"])
    console.print(table)

    while True:
        choice = Prompt.ask(
            "Pick a provider (number or key)",
            default="ollama",
        ).strip()
        if choice.isdigit():
            i = int(choice)
            if 1 <= i <= len(PROVIDER_ORDER):
                return PROVIDER_ORDER[i - 1]
        elif choice in PROVIDER_PRESETS:
            return choice
        console.print(f"[red]Not a valid choice: {choice}[/]")


def _resolve_base_url(
    console: Console, provider_key: str, preset: dict
) -> str | None:
    if preset["kind"] != "openai_compat":
        return None
    default = preset.get("base_url")
    prompt = "Base URL"
    if default:
        return Prompt.ask(prompt, default=default).strip()
    return Prompt.ask(prompt).strip()


def _resolve_api_key_env(
    console: Console, provider_key: str, preset: dict
) -> str | None:
    """For cloud providers we record the env var *name* (never the key itself)."""
    env_name = preset.get("api_key_env")
    if not env_name:
        return None
    if not os.getenv(env_name):
        console.print(
            f"[yellow]Heads up:[/] [bold]{env_name}[/] is not set in your environment. "
            f"Add it to your shell or .env before running boardroom.py."
        )
    return env_name


async def _pick_model(
    console: Console,
    provider_key: str,
    base_url: str | None,
    preset: dict,
    api_key_env: str | None,
) -> str:
    """Try to discover models from the provider; fall back to free-text."""
    discovered: list[str] = []

    # Try discovery for cloud providers only when their key is actually present;
    # otherwise list_models() will fail with an auth error.
    can_discover = preset["kind"] == "openai_compat" and (
        api_key_env is None or os.getenv(api_key_env)
    )
    # For Anthropic we always have a curated list, no live discovery needed.
    if preset["kind"] == "anthropic":
        can_discover = True

    if can_discover:
        try:
            api_key = os.getenv(api_key_env) if api_key_env else preset.get("api_key", "not-needed")
            provider = make_provider(provider_key, base_url=base_url, api_key=api_key)
            try:
                discovered = await provider.list_models()
            finally:
                await provider.close()
        except Exception as e:
            console.print(f"[yellow]Could not list models from {provider_key}:[/] {e}")
            console.print(
                f"[yellow]Tip:[/] make sure the server is running"
                f"{f' at {base_url}' if base_url else ''}."
            )

    if discovered:
        console.print()
        console.print("[bold]Available models:[/]")
        for i, name in enumerate(discovered, 1):
            console.print(f"  [dim]{i:>2}.[/] {name}")
        while True:
            answer = Prompt.ask(
                "Pick a model (number or full name)",
                default=discovered[0],
            ).strip()
            if answer.isdigit():
                i = int(answer)
                if 1 <= i <= len(discovered):
                    return discovered[i - 1]
                console.print(f"[red]Out of range: {i}[/]")
                continue
            return answer

    return Prompt.ask("Model name").strip()


def _write_config(
    console: Console,
    config_path: Path,
    *,
    provider: str,
    model: str,
    base_url: str | None,
    api_key_env: str | None,
) -> None:
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text()) or {}
    else:
        data = {}

    defaults = data.get("defaults") or {}
    defaults["provider"] = provider
    defaults["model"] = model
    defaults.setdefault("max_tokens", 600)
    if base_url:
        defaults["base_url"] = base_url
    else:
        defaults.pop("base_url", None)
    if api_key_env:
        defaults["api_key_env"] = api_key_env
    else:
        defaults.pop("api_key_env", None)
    data["defaults"] = defaults

    if "agents" not in data:
        console.print(
            "[yellow]Note:[/] no `agents:` section found in agents.yaml; "
            "personas must be defined before running a discussion."
        )

    if config_path.exists() and not Confirm.ask(
        f"Overwrite [bold]{config_path}[/] (comments in `defaults:` may be lost)?",
        default=True,
    ):
        console.print("[red]Aborted.[/] No changes written.")
        return

    config_path.write_text(yaml.safe_dump(data, sort_keys=False, indent=2))
    console.print(f"[green]Wrote[/] {config_path}")
