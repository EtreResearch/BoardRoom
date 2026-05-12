"""Textual TUI for BoardRoom.

Layout: header · (chat log + in-flight panel | sidebar with roster + tally) · footer.
A background worker iterates the engine's event stream and updates widgets.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from anthropic import AsyncAnthropic
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, RichLog, Static

from engine import (
    Agent,
    Error,
    RoundStart,
    TallyComplete,
    Token,
    Turn,
    TurnEnd,
    TurnStart,
    UsageReport,
    Verdict,
    VerdictRoundStart,
    compute_cost,
    run_boardroom,
)

STATUS_LABELS = {
    "pending": "[dim]pending[/]",
    "speaking": "[yellow]speaking…[/]",
    "done": "[dim]done[/]",
    "voted_good": "[green]voted GOOD[/]",
    "voted_bad": "[red]voted BAD[/]",
    "voted_unclear": "[yellow]voted ?[/]",
}


class AgentRoster(Vertical):
    """Sidebar widget showing each agent's current status."""

    def __init__(self, agents: list[Agent]) -> None:
        super().__init__(id="roster")
        self.agents = agents
        self.statuses: dict[str, str] = {a.role: "pending" for a in agents}

    def compose(self) -> ComposeResult:
        yield Static("[b]Agents[/]", classes="section-title")
        for agent in self.agents:
            yield Static(
                self._render(agent),
                id=f"role-{self._slug(agent.role)}",
                classes="roster-row",
            )

    def update_status(self, role: str, status: str) -> None:
        self.statuses[role] = status
        agent = next(a for a in self.agents if a.role == role)
        widget = self.query_one(f"#role-{self._slug(role)}", Static)
        widget.update(self._render(agent))

    def _render(self, agent: Agent) -> str:
        status = self.statuses[agent.role]
        return f"[bold {agent.color}]●[/] {agent.role:<14} {STATUS_LABELS[status]}"

    @staticmethod
    def _slug(role: str) -> str:
        return role.lower().replace(" ", "-")


class TallySummary(Static):
    """Sidebar widget showing live and final vote tally."""

    def __init__(self) -> None:
        super().__init__("", id="tally")
        self.good = 0
        self.bad = 0
        self.overall: str | None = None
        self.update(self._render())

    def add_vote(self, verdict: str) -> None:
        if verdict == "GOOD":
            self.good += 1
        elif verdict == "BAD":
            self.bad += 1
        self.update(self._render())

    def finalize(self, good: int, bad: int, overall: str) -> None:
        self.good = good
        self.bad = bad
        self.overall = overall
        self.update(self._render())

    def _render(self) -> str:
        body = f"{self.good} GOOD / {self.bad} BAD"
        if self.overall is None:
            return f"[b]Tally[/]\n\n{body}"
        styled = {
            "GOOD": "[bold green]GOOD[/]",
            "BAD": "[bold red]BAD[/]",
            "SPLIT": "[bold yellow]SPLIT[/]",
        }[self.overall]
        return f"[b]Tally[/]\n\n{body}\n\nVerdict: {styled}"


class TokenUsage(Static):
    """Sidebar widget showing live token usage and estimated cost."""

    def __init__(self) -> None:
        super().__init__("", id="token-usage")
        self.total_input = 0
        self.total_output = 0
        self.total_cache_write = 0
        self.total_cache_read = 0
        self.total_cost = 0.0
        self.last_role: str | None = None
        self.last_input = 0
        self.last_output = 0
        self.update(self._render())

    def add_usage(self, event: UsageReport) -> None:
        self.total_input += event.input_tokens
        self.total_output += event.output_tokens
        self.total_cache_write += event.cache_creation_input_tokens
        self.total_cache_read += event.cache_read_input_tokens
        self.total_cost += compute_cost(event)
        self.last_role = event.role
        # "Input" shown to the user includes uncached + cache write + cache read,
        # since they all count as bytes the model saw.
        self.last_input = (
            event.input_tokens
            + event.cache_creation_input_tokens
            + event.cache_read_input_tokens
        )
        self.last_output = event.output_tokens
        self.update(self._render())

    def _render(self) -> str:
        if self.last_role is None:
            return "[b]Usage[/]\n\n[dim]—[/]"
        total_input = self.total_input + self.total_cache_write + self.total_cache_read
        body = (
            f"[b]Usage[/]  [dim]~${self.total_cost:.4f}[/]\n\n"
            f"[dim]Last ({self.last_role}):[/] "
            f"{self.last_input:,} in · {self.last_output:,} out\n"
            f"[dim]Session:[/] "
            f"{total_input:,} in · {self.total_output:,} out"
        )
        if self.total_cache_read or self.total_cache_write:
            body += (
                f"\n[dim]Cache:[/] "
                f"{self.total_cache_write:,} w · {self.total_cache_read:,} r"
            )
        return body


class SetupScreen(ModalScreen[bool]):
    """One-question setup picker shown before the discussion starts."""

    BINDINGS = [
        ("q", "app.quit", "Quit"),
        ("s", "pick_shuffled", "Shuffled"),
        ("t", "pick_structured", "Structured"),
    ]

    def __init__(self, default_ordered: bool) -> None:
        super().__init__()
        self.default_ordered = default_ordered

    def compose(self) -> ComposeResult:
        with Container(id="setup-container"):
            yield Static("[b]Setup[/]", id="setup-title")
            yield Static("How should the executives take turns?", id="setup-question")
            with Horizontal(id="setup-options"):
                yield Button(
                    "Shuffled",
                    id="opt-shuffled",
                    variant="primary" if not self.default_ordered else "default",
                )
                yield Button(
                    "Structured",
                    id="opt-structured",
                    variant="primary" if self.default_ordered else "default",
                )
            yield Static(
                "[dim]Tab / ← → toggle · Enter accept · q quit[/]",
                id="setup-help",
            )

    def on_mount(self) -> None:
        focus_id = "opt-structured" if self.default_ordered else "opt-shuffled"
        self.query_one(f"#{focus_id}", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "opt-structured")

    def action_pick_shuffled(self) -> None:
        self.dismiss(False)

    def action_pick_structured(self) -> None:
        self.dismiss(True)


class BoardRoomApp(App):
    """Live boardroom discussion in a Textual TUI."""

    CSS_PATH = "boardroom.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "save_transcript", "Save transcript"),
    ]

    def __init__(
        self,
        idea: str,
        agents: list[Agent],
        rounds: int,
        model: str,
        max_tokens: int,
        ordered: bool = False,
        seed: int | None = None,
        no_setup: bool = False,
    ) -> None:
        super().__init__()
        self.idea = idea
        self.agents = agents
        self.rounds = rounds
        self.model = model
        self.max_tokens = max_tokens
        self.ordered = ordered
        self.seed = seed
        self.no_setup = no_setup
        self._buffer = ""
        self._transcript: list[Turn] = []
        self._in_verdict_round = False
        self._client: AsyncAnthropic | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main"):
            with Vertical(id="chat-column"):
                yield RichLog(id="chat", markup=True, wrap=True, highlight=False)
                yield Static("", id="current")
            with Vertical(id="sidebar"):
                yield AgentRoster(self.agents)
                yield TallySummary()
                yield TokenUsage()
        yield Footer()

    def on_mount(self) -> None:
        self.title = "BoardRoom"
        idea_short = self.idea if len(self.idea) <= 80 else self.idea[:79] + "…"
        self.sub_title = idea_short
        if self.no_setup:
            self._run_discussion()
        else:
            self.push_screen(SetupScreen(self.ordered), self._on_setup_done)

    def _on_setup_done(self, ordered: bool | None) -> None:
        if ordered is not None:
            self.ordered = ordered
        self._run_discussion()

    async def on_unmount(self) -> None:
        if self._client is not None:
            await self._client.close()

    @work(exclusive=True, group="discussion")
    async def _run_discussion(self) -> None:
        self._client = AsyncAnthropic()
        chat = self.query_one("#chat", RichLog)
        current = self.query_one("#current", Static)
        roster = self.query_one(AgentRoster)
        tally = self.query_one(TallySummary)
        usage_widget = self.query_one(TokenUsage)

        async for event in run_boardroom(
            self._client,
            self.agents,
            self.idea,
            self.rounds,
            self.model,
            self.max_tokens,
            ordered=self.ordered,
            seed=self.seed,
        ):
            if isinstance(event, RoundStart):
                order_text = " → ".join(event.order)
                chat.write(
                    Text(
                        f"── Round {event.n} of {event.total}  ·  "
                        f"{order_text} ──",
                        style="dim",
                    )
                )

            elif isinstance(event, VerdictRoundStart):
                self._in_verdict_round = True
                chat.write("[dim]── Final verdicts ──[/]")
                # Reset roster to pending for the verdict round
                for agent in self.agents:
                    roster.update_status(agent.role, "pending")

            elif isinstance(event, TurnStart):
                self._buffer = ""
                color = self._color(event.role)
                current.update(
                    Text.assemble(
                        (f"{event.role}: ", f"bold {color}"),
                        ("…", "dim"),
                    )
                )
                roster.update_status(event.role, "speaking")

            elif isinstance(event, Token):
                self._buffer += event.text
                color = self._color(event.role)
                current.update(
                    Text.assemble(
                        (f"{event.role}: ", f"bold {color}"),
                        Text(self._buffer),
                    )
                )

            elif isinstance(event, TurnEnd):
                color = self._color(event.role)
                chat.write(
                    Text.assemble(
                        (f"{event.role}: ", f"bold {color}"),
                        Text(event.full_text.strip()),
                    )
                )
                chat.write("")
                current.update("")
                self._transcript.append(Turn(speaker=event.role, text=event.full_text.strip()))
                if not self._in_verdict_round:
                    roster.update_status(event.role, "done")

            elif isinstance(event, Verdict):
                status = {
                    "GOOD": "voted_good",
                    "BAD": "voted_bad",
                    "UNCLEAR": "voted_unclear",
                }[event.verdict]
                roster.update_status(event.role, status)
                tally.add_vote(event.verdict)

            elif isinstance(event, TallyComplete):
                tally.finalize(event.good, event.bad, event.overall)
                styled = {
                    "GOOD": "[bold green]GOOD[/]",
                    "BAD": "[bold red]BAD[/]",
                    "SPLIT": "[bold yellow]SPLIT[/]",
                }[event.overall]
                chat.write(
                    f"[b]Final verdict:[/] {styled}  "
                    f"[dim]({event.good} GOOD / {event.bad} BAD)[/]"
                )

            elif isinstance(event, UsageReport):
                usage_widget.add_usage(event)

            elif isinstance(event, Error):
                self.notify(f"{event.role or 'engine'}: {event.message}", severity="error")

    def _color(self, role: str) -> str:
        for a in self.agents:
            if a.role == role:
                return a.color
        return "white"

    def action_save_transcript(self) -> None:
        if not self._transcript:
            self.notify("Nothing to save yet.")
            return
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(f"transcript-{ts}.md")
        lines = ["# BoardRoom transcript", "", f"**Idea:** {self.idea}", ""]
        for turn in self._transcript:
            lines += [f"## {turn.speaker}", "", turn.text, ""]
        path.write_text("\n".join(lines))
        self.notify(f"Saved {path.name}")
