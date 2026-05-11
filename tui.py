"""Textual TUI for BoardRoom — focus mode.

One centered card shows only the current speaker streaming live. A compact
status row at the bottom tracks all four agents (○ pending, ◐ speaking,
● done, ✓ voted GOOD, ✗ voted BAD) plus the running tally. Press `h` to
toggle a scrollable history overlay; the live discussion keeps running
underneath.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from anthropic import AsyncAnthropic
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Center, Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Static

from engine import (
    Agent,
    Error,
    RoundStart,
    TallyComplete,
    Token,
    Turn,
    TurnEnd,
    TurnStart,
    Verdict,
    VerdictRoundStart,
    run_boardroom,
)


def _slug(role: str) -> str:
    return role.lower().replace(" ", "-")


class FocalCard(Static):
    """The single centered card showing the current speaker."""

    def __init__(self) -> None:
        super().__init__("", id="focal")
        self.border_title = " — "

    def show_idle(self) -> None:
        self.border_title = " Preparing "
        self.styles.border = ("round", "grey50")
        self.update(Text("Waiting for the discussion to begin…", style="italic dim"))

    def set_speaker(self, role: str, color: str) -> None:
        self.border_title = f"  ●  {role}  "
        self.styles.border = ("round", color)
        self.update(Text(""))

    def show_text(self, text: str) -> None:
        self.update(Text(text))

    def show_verdict(self, overall: str, good: int, bad: int) -> None:
        color = {"GOOD": "bright_green", "BAD": "bright_red", "SPLIT": "yellow"}[overall]
        self.border_title = "  Final verdict  "
        self.styles.border = ("heavy", color)
        body = Text()
        body.append("\n")
        body.append(f"  {overall}  \n", style=f"bold {color}")
        body.append("\n")
        body.append(f"Tally: {good} GOOD · {bad} BAD", style="dim")
        self.update(body)


class StatusRow(Container):
    """Single-line status indicator showing every agent + running tally."""

    def __init__(self, agents: list[Agent]) -> None:
        super().__init__(id="status-row")
        self.agents = agents
        self.statuses: dict[str, str] = {a.role: "pending" for a in agents}
        self.good = 0
        self.bad = 0

    def compose(self) -> ComposeResult:
        for agent in self.agents:
            yield Static(
                self._render_agent(agent),
                id=f"dot-{_slug(agent.role)}",
                classes="status-dot",
            )
        yield Static(self._render_tally(), id="status-tally")

    def update_agent(self, role: str, status: str) -> None:
        self.statuses[role] = status
        agent = next(a for a in self.agents if a.role == role)
        self.query_one(f"#dot-{_slug(role)}", Static).update(self._render_agent(agent))

    def add_vote(self, verdict: str) -> None:
        if verdict == "GOOD":
            self.good += 1
        elif verdict == "BAD":
            self.bad += 1
        self.query_one("#status-tally", Static).update(self._render_tally())

    def reset_for_verdicts(self) -> None:
        for a in self.agents:
            self.statuses[a.role] = "pending"
            self.query_one(f"#dot-{_slug(a.role)}", Static).update(self._render_agent(a))

    def _render_agent(self, agent: Agent) -> str:
        status = self.statuses[agent.role]
        if status == "pending":
            return f"[grey50]○ {agent.role}[/]"
        if status == "speaking":
            return f"[bold {agent.color}]◐ {agent.role}[/]"
        if status == "done":
            return f"[{agent.color}]● {agent.role}[/]"
        if status == "voted_good":
            return f"[green]✓ {agent.role}[/]"
        if status == "voted_bad":
            return f"[red]✗ {agent.role}[/]"
        return f"[yellow]? {agent.role}[/]"

    def _render_tally(self) -> str:
        return f"[green]{self.good} GOOD[/] · [red]{self.bad} BAD[/]"


class HistoryScreen(ModalScreen):
    """Scrollable overlay listing every past turn. Live discussion keeps going underneath."""

    BINDINGS = [
        ("h", "dismiss", "Close"),
        ("escape", "dismiss", "Close"),
    ]

    def __init__(self, transcript: list[Turn], color_for) -> None:
        super().__init__()
        self.transcript = transcript
        self.color_for = color_for

    def compose(self) -> ComposeResult:
        with Container(id="history-container"):
            yield Static(
                "[b]Discussion history[/]   [dim](press h or esc to return — live continues)[/]",
                id="history-title",
            )
            with VerticalScroll(id="history-scroll"):
                if not self.transcript:
                    yield Static("[dim italic]No turns yet.[/]", classes="history-empty")
                else:
                    for i, turn in enumerate(self.transcript, 1):
                        color = self.color_for(turn.speaker)
                        body = Text()
                        body.append(f"{i}. {turn.speaker}\n", style=f"bold {color}")
                        body.append(turn.text)
                        yield Static(body, classes="history-turn")


class BoardRoomApp(App):
    """Live boardroom discussion in focus mode."""

    CSS_PATH = "boardroom.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("h", "toggle_history", "History"),
        ("s", "save_transcript", "Save"),
    ]

    def __init__(
        self,
        idea: str,
        agents: list[Agent],
        rounds: int,
        model: str,
        max_tokens: int,
    ) -> None:
        super().__init__()
        self.idea = idea
        self.agents = agents
        self.rounds = rounds
        self.model = model
        self.max_tokens = max_tokens
        self._buffer = ""
        self._transcript: list[Turn] = []
        self._in_verdict_round = False
        self._round = 0
        self._client: AsyncAnthropic | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("", id="sub-header")
        with Center(id="stage"):
            yield FocalCard()
        yield StatusRow(self.agents)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "BoardRoom"
        self.sub_title = self.idea if len(self.idea) <= 80 else self.idea[:79] + "…"
        self.query_one(FocalCard).show_idle()
        self._set_phase_text()
        self._run_discussion()

    async def on_unmount(self) -> None:
        if self._client is not None:
            await self._client.close()

    def _set_phase_text(self) -> None:
        sub = self.query_one("#sub-header", Static)
        if self._round == 0:
            sub.update("[dim italic]Preparing discussion…[/]")
        elif self._in_verdict_round:
            sub.update("[dim]Final verdicts[/]")
        else:
            sub.update(f"[dim]Round {self._round} of {self.rounds}[/]")

    @work(exclusive=True, group="discussion")
    async def _run_discussion(self) -> None:
        self._client = AsyncAnthropic()
        focal = self.query_one(FocalCard)
        status = self.query_one(StatusRow)

        async for event in run_boardroom(
            self._client,
            self.agents,
            self.idea,
            self.rounds,
            self.model,
            self.max_tokens,
        ):
            if isinstance(event, RoundStart):
                self._round = event.n
                self._set_phase_text()

            elif isinstance(event, VerdictRoundStart):
                self._in_verdict_round = True
                status.reset_for_verdicts()
                self._set_phase_text()

            elif isinstance(event, TurnStart):
                self._buffer = ""
                color = self._color(event.role)
                focal.set_speaker(event.role, color)
                status.update_agent(event.role, "speaking")

            elif isinstance(event, Token):
                self._buffer += event.text
                focal.show_text(self._buffer)

            elif isinstance(event, TurnEnd):
                focal.show_text(event.full_text.strip())
                self._transcript.append(
                    Turn(speaker=event.role, text=event.full_text.strip())
                )
                if not self._in_verdict_round:
                    status.update_agent(event.role, "done")

            elif isinstance(event, Verdict):
                key = {
                    "GOOD": "voted_good",
                    "BAD": "voted_bad",
                    "UNCLEAR": "voted_unclear",
                }[event.verdict]
                status.update_agent(event.role, key)
                status.add_vote(event.verdict)

            elif isinstance(event, TallyComplete):
                focal.show_verdict(event.overall, event.good, event.bad)
                self.query_one("#sub-header", Static).update("[bold]Decision[/]")

            elif isinstance(event, Error):
                self.notify(
                    f"{event.role or 'engine'}: {event.message}", severity="error"
                )

    def _color(self, role: str) -> str:
        for a in self.agents:
            if a.role == role:
                return a.color
        return "white"

    def action_toggle_history(self) -> None:
        # If a modal is already on top of the stack, dismiss it; otherwise push.
        if len(self.screen_stack) > 1:
            self.pop_screen()
        else:
            self.push_screen(HistoryScreen(list(self._transcript), self._color))

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
