"""Textual TUI for BoardRoom.

Layout: header · (chat log + in-flight panel | sidebar with roster + tally) · footer.
A background worker iterates the engine's event stream and updates widgets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml
from anthropic import AsyncAnthropic
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, RichLog, Static

from engine import (
    DEFAULT_VERDICT_MODE,
    Agent,
    DecisionFrame,
    DecisionFrameStart,
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
    """Sidebar widget showing live and final vote tally.

    During the verdict round, displays a running raw count. On
    `finalize()` (driven by the `TallyComplete` event), swaps to a
    stratified view that shows conviction strength so a weak GOOD can't
    visually masquerade as a strong GOOD.
    """

    def __init__(self) -> None:
        super().__init__("", id="tally")
        self.good = 0
        self.bad = 0
        self.overall: str | None = None
        # Stratified fields populated on finalize().
        self.strong_good = 0
        self.lean_good = 0
        self.weak_good = 0
        self.strong_bad = 0
        self.lean_bad = 0
        self.weak_bad = 0
        self.unclear = 0
        self.net_score = 0.0
        self.headline = ""
        self.strong_dissent = 0
        self.update(self._render())

    def add_vote(self, verdict: str) -> None:
        if verdict == "GOOD":
            self.good += 1
        elif verdict == "BAD":
            self.bad += 1
        self.update(self._render())

    def finalize(self, event: "TallyComplete") -> None:
        self.good = event.good
        self.bad = event.bad
        self.overall = event.overall
        self.strong_good = event.strong_good
        self.lean_good = event.lean_good
        self.weak_good = event.weak_good
        self.strong_bad = event.strong_bad
        self.lean_bad = event.lean_bad
        self.weak_bad = event.weak_bad
        self.unclear = event.unclear
        self.net_score = event.net_score
        self.headline = event.headline
        self.strong_dissent = event.strong_dissent
        self.update(self._render())

    @staticmethod
    def _headline_color(headline: str) -> str:
        if "GOOD" in headline:
            return "green"
        if "BAD" in headline:
            return "red"
        return "yellow"

    def _render(self) -> str:
        # Live view (during verdict round, before TallyComplete).
        if self.overall is None:
            return f"[b]Tally[/]\n\n{self.good} GOOD · {self.bad} BAD"

        # `--verdict simple` produces a finalized TallyComplete with an
        # empty headline. Render the legacy count + overall rather than
        # the stratified view it doesn't have data for.
        if not self.headline:
            styled = {
                "GOOD": "[bold green]GOOD[/]",
                "BAD": "[bold red]BAD[/]",
                "SPLIT": "[bold yellow]SPLIT[/]",
            }[self.overall]
            return (
                f"[b]Tally[/]\n\n{self.good} GOOD / {self.bad} BAD"
                f"\n\nVerdict: {styled}"
            )

        # Finalized stratified view.
        color = self._headline_color(self.headline)
        warning = (
            f"  [yellow]⚠ {self.strong_dissent} strong dissent[/]"
            if self.strong_dissent
            else ""
        )

        lines = [
            "[b]Tally[/]",
            "",
            f"[bold {color}]{self.headline}[/]{warning}",
            "",
        ]
        good_total = self.strong_good + self.lean_good + self.weak_good
        bad_total = self.strong_bad + self.lean_bad + self.weak_bad
        if good_total:
            lines.append(
                f"[green]GOOD[/]  {self.strong_good} strong · "
                f"{self.lean_good} lean"
            )
        if bad_total:
            lines.append(
                f"[red]BAD[/]   {self.strong_bad} strong · "
                f"{self.lean_bad} lean"
            )
        weak_total = self.weak_good + self.weak_bad
        if weak_total:
            lines.append(f"[dim]{weak_total} weak[/]")
        if self.unclear:
            lines.append(f"[yellow]{self.unclear} unclear[/]")
        lines.append("")
        lines.append(f"[dim]Weighted: {self.net_score:+.0%}[/]")
        return "\n".join(lines)


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


class InterjectScreen(ModalScreen[str | None]):
    """Modal for capturing a one-shot user directive for the next round."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Container(id="interject-container"):
            yield Static("[b]Steer the discussion[/]", id="interject-title")
            yield Static(
                "Your directive is attached to every agent's prompt in the "
                "next round (or the verdict round if this is the last one). "
                "The current round keeps streaming uninterrupted.",
                id="interject-help",
            )
            yield Input(
                placeholder="e.g. focus on privacy concerns",
                id="interject-input",
            )
            yield Static("[dim]Enter to submit · Esc to cancel[/]", id="interject-hint")

    def on_mount(self) -> None:
        self.query_one("#interject-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.dismiss(text or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class BoardRoomApp(App):
    """Live boardroom discussion in a Textual TUI."""

    CSS_PATH = "boardroom.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "save_transcript", "Save transcript"),
        ("i", "interject", "Interject"),
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
        verdict_mode: str = DEFAULT_VERDICT_MODE,
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
        self.verdict_mode = verdict_mode
        self._buffer = ""
        self._transcript: list[Turn] = []
        self._in_verdict_round = False
        self._client: AsyncAnthropic | None = None
        # Session metadata captured for the saved transcript.
        self._rounds_data: list[dict] = []          # [{n, speaking_order, directive, turns: [...]}]
        self._verdicts: list[dict] = []             # [{role, verdict, confidence, reasoning, disconfirming}]
        self._tally: dict | None = None              # {good, bad, overall, ...strata}
        self._decision_frame: dict | None = None     # {case_for, case_against, biggest_unknown, conditions}
        self._verdict_directive: str | None = None   # directive applied to the verdict round
        # Interjection state.
        self._pending_directive: str | None = None
        self._past_discussion: bool = False         # True once VerdictRoundStart fires

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
            consume_directive=self._consume_directive,
            verdict_mode=self.verdict_mode,
        ):
            if isinstance(event, RoundStart):
                self._rounds_data.append(
                    {
                        "n": event.n,
                        "speaking_order": list(event.order),
                        "directive": event.directive,
                        "turns": [],
                    }
                )
                order_text = " → ".join(event.order)
                chat.write(
                    Text(
                        f"── Round {event.n} of {event.total}  ·  "
                        f"{order_text} ──",
                        style="dim",
                    )
                )
                if event.directive:
                    chat.write(
                        Text(
                            f"  User directive: {event.directive}",
                            style="italic dim",
                        )
                    )

            elif isinstance(event, VerdictRoundStart):
                self._in_verdict_round = True
                self._past_discussion = True
                self._verdict_directive = event.directive
                chat.write("[dim]── Final verdicts ──[/]")
                if event.directive:
                    chat.write(
                        Text(
                            f"  User directive: {event.directive}",
                            style="italic dim",
                        )
                    )
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
                text = event.full_text.strip()
                chat.write(
                    Text.assemble(
                        (f"{event.role}: ", f"bold {color}"),
                        Text(text),
                    )
                )
                chat.write("")
                current.update("")
                self._transcript.append(Turn(speaker=event.role, text=text))
                if not self._in_verdict_round:
                    if self._rounds_data:
                        self._rounds_data[-1]["turns"].append(
                            {"speaker": event.role, "text": text}
                        )
                    roster.update_status(event.role, "done")

            elif isinstance(event, Verdict):
                status = {
                    "GOOD": "voted_good",
                    "BAD": "voted_bad",
                    "UNCLEAR": "voted_unclear",
                }[event.verdict]
                roster.update_status(event.role, status)
                tally.add_vote(event.verdict)
                self._verdicts.append(
                    {
                        "role": event.role,
                        "verdict": event.verdict,
                        "confidence": event.confidence,
                        "reasoning": event.reasoning or event.text,
                        "disconfirming": event.disconfirming,
                    }
                )

            elif isinstance(event, TallyComplete):
                tally.finalize(event)
                self._tally = {
                    "good": event.good,
                    "bad": event.bad,
                    "overall": event.overall,
                    "headline": event.headline,
                    "net_score": round(event.net_score, 4),
                    "strong_good": event.strong_good,
                    "lean_good": event.lean_good,
                    "weak_good": event.weak_good,
                    "strong_bad": event.strong_bad,
                    "lean_bad": event.lean_bad,
                    "weak_bad": event.weak_bad,
                    "unclear": event.unclear,
                    "strong_dissent": event.strong_dissent,
                }
                if event.headline:
                    headline_color = (
                        "green" if "GOOD" in event.headline
                        else "red" if "BAD" in event.headline
                        else "yellow"
                    )
                    warning = (
                        f"  [yellow]⚠ {event.strong_dissent} strong dissent[/]"
                        if event.strong_dissent
                        else ""
                    )
                    chat.write(
                        f"[b]Final verdict:[/] "
                        f"[bold {headline_color}]{event.headline}[/]"
                        f"{warning}  "
                        f"[dim]({event.good} GOOD / {event.bad} BAD · "
                        f"weighted {event.net_score:+.0%})[/]"
                    )
                else:
                    # `--verdict simple`: no confidence-stratified data;
                    # fall back to the legacy line.
                    color = {
                        "GOOD": "green", "BAD": "red", "SPLIT": "yellow",
                    }[event.overall]
                    chat.write(
                        f"[b]Final verdict:[/] [bold {color}]{event.overall}[/]"
                        f"  [dim]({event.good} GOOD / {event.bad} BAD)[/]"
                    )

            elif isinstance(event, DecisionFrameStart):
                chat.write(Text("── Decision frame ──", style="dim"))
                chat.write(Text("Synthesizing…", style="italic dim"))

            elif isinstance(event, DecisionFrame):
                # Capture for the saved transcript.
                self._decision_frame = {
                    "case_for": event.case_for,
                    "case_against": event.case_against,
                    "biggest_unknown": event.biggest_unknown,
                    "conditions": list(event.conditions),
                }
                if event.case_for:
                    chat.write(
                        Text.assemble(
                            ("Case for: ", "bold green"),
                            Text(event.case_for),
                        )
                    )
                    chat.write("")
                if event.case_against:
                    chat.write(
                        Text.assemble(
                            ("Case against: ", "bold red"),
                            Text(event.case_against),
                        )
                    )
                    chat.write("")
                if event.biggest_unknown:
                    chat.write(
                        Text.assemble(
                            ("Biggest unknown: ", "bold yellow"),
                            Text(event.biggest_unknown),
                        )
                    )
                    chat.write("")
                if event.conditions:
                    chat.write(Text("Conditions for proceeding:", style="bold"))
                    for c in event.conditions:
                        chat.write(Text.assemble(("  • ", "bold"), Text(c)))

            elif isinstance(event, UsageReport):
                usage_widget.add_usage(event)

            elif isinstance(event, Error):
                self.notify(f"{event.role or 'engine'}: {event.message}", severity="error")

    def _color(self, role: str) -> str:
        for a in self.agents:
            if a.role == role:
                return a.color
        return "white"

    def _build_transcript_document(self) -> str:
        """Render the full session as a Markdown doc with YAML frontmatter."""
        usage = self.query_one(TokenUsage)
        saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        meta = {
            "saved_at": saved_at,
            "idea": self.idea,
            "config": {
                "model": self.model,
                "rounds": self.rounds,
                "max_tokens": self.max_tokens,
                "ordered": self.ordered,
                "seed": self.seed,
                "verdict_mode": self.verdict_mode,
            },
            "agents": [
                {"role": a.role, "color": a.color, "system": a.system}
                for a in self.agents
            ],
            "rounds": self._rounds_data,
            "verdict_directive": self._verdict_directive,
            "verdicts": self._verdicts,
            "tally": self._tally,
            "decision_frame": self._decision_frame,
            "usage": {
                "input_tokens": usage.total_input,
                "output_tokens": usage.total_output,
                "cache_creation_input_tokens": usage.total_cache_write,
                "cache_read_input_tokens": usage.total_cache_read,
                "estimated_cost_usd": round(usage.total_cost, 6),
            },
        }
        frontmatter = yaml.safe_dump(
            meta,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

        order_label = "ordered" if self.ordered else "shuffled"
        header = (
            f"# BoardRoom transcript\n\n"
            f"**Idea:** {self.idea}\n\n"
            f"**Model:** {self.model} · **Rounds:** {self.rounds} · "
            f"**Order:** {order_label}"
        )
        if self.seed is not None:
            header += f" · **Seed:** {self.seed}"
        header += f"  \n**Saved:** {saved_at}\n"

        body: list[str] = [f"---\n{frontmatter}---\n", header]

        for r in self._rounds_data:
            body.append(f"\n## Round {r['n']} of {self.rounds}\n")
            order_text = " → ".join(r["speaking_order"])
            body.append(f"_Speaking order: {order_text}_\n")
            if r.get("directive"):
                body.append(f"_User directive: {r['directive']}_\n")
            for t in r["turns"]:
                body.append(f"\n### {t['speaker']}\n\n{t['text']}\n")

        if self._verdicts:
            body.append("\n## Verdicts\n")
            if self._verdict_directive:
                body.append(f"_User directive: {self._verdict_directive}_\n")
            for v in self._verdicts:
                conf_str = f" (conf {v['confidence']}/5)" if v.get("confidence") else ""
                body.append(
                    f"- **{v['role']}** — **{v['verdict']}**{conf_str} — "
                    f"_{v['reasoning']}_"
                )
                if v.get("disconfirming"):
                    body.append(
                        f"  - _Strongest reason I might be wrong:_ {v['disconfirming']}"
                    )

        if self._tally is not None:
            t = self._tally
            if not t.get("headline"):
                # `--verdict simple`: no strata data; render legacy.
                body.append(
                    f"\n## Tally\n\n"
                    f"**{t['good']} GOOD / {t['bad']} BAD → {t['overall']}**\n"
                )
            else:
                warning = (
                    f"  ⚠ {t['strong_dissent']} strong dissent"
                    if t.get("strong_dissent")
                    else ""
                )
                tally_lines = [
                    f"\n## Tally\n",
                    f"**{t['headline']}**{warning}",
                    "",
                ]
                good_total = t.get("strong_good", 0) + t.get("lean_good", 0) + t.get("weak_good", 0)
                bad_total = t.get("strong_bad", 0) + t.get("lean_bad", 0) + t.get("weak_bad", 0)
                if good_total:
                    tally_lines.append(
                        f"- GOOD: {t.get('strong_good', 0)} strong · "
                        f"{t.get('lean_good', 0)} lean"
                    )
                if bad_total:
                    tally_lines.append(
                        f"- BAD: {t.get('strong_bad', 0)} strong · "
                        f"{t.get('lean_bad', 0)} lean"
                    )
                weak_total = t.get("weak_good", 0) + t.get("weak_bad", 0)
                if weak_total:
                    tally_lines.append(f"- Weak: {weak_total}")
                if t.get("unclear"):
                    tally_lines.append(f"- Unclear: {t['unclear']}")
                tally_lines.append("")
                net = t.get("net_score", 0.0)
                tally_lines.append(
                    f"_Raw count: {t['good']} GOOD / {t['bad']} BAD · "
                    f"weighted {net:+.0%}_\n"
                )
                body.append("\n".join(tally_lines))

        if self._decision_frame is not None:
            df = self._decision_frame
            df_lines = ["\n## Decision frame\n"]
            if df.get("case_for"):
                df_lines.append(f"**Case for:** {df['case_for']}\n")
            if df.get("case_against"):
                df_lines.append(f"**Case against:** {df['case_against']}\n")
            if df.get("biggest_unknown"):
                df_lines.append(f"**Biggest unknown:** {df['biggest_unknown']}\n")
            if df.get("conditions"):
                df_lines.append("**Conditions for proceeding:**\n")
                for c in df["conditions"]:
                    df_lines.append(f"- {c}")
                df_lines.append("")
            body.append("\n".join(df_lines))

        if usage.last_role is not None:
            # Match the TUI sidebar: "Input" is the full count the model saw
            # (uncached + cache write + cache read). The "Cache" line below
            # breaks down which slice of that came from the cache.
            total_input = (
                usage.total_input
                + usage.total_cache_write
                + usage.total_cache_read
            )
            body.append(
                "\n## Usage\n\n"
                f"- Input: {total_input:,} tokens (incl. cache)\n"
                f"- Output: {usage.total_output:,} tokens\n"
                f"- Cache write: {usage.total_cache_write:,} · "
                f"Cache read: {usage.total_cache_read:,}\n"
                f"- Estimated cost: ${usage.total_cost:.4f}\n"
            )

        return "\n".join(body)

    def action_save_transcript(self) -> None:
        if not self._transcript:
            self.notify("Nothing to save yet.")
            return
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(f"transcript-{ts}.md")
        path.write_text(self._build_transcript_document())
        self.notify(f"Saved {path.name}")

    def action_interject(self) -> None:
        if self._past_discussion:
            self.notify(
                "Discussion is past the discussion rounds; directive ignored."
            )
            return
        self.push_screen(InterjectScreen(), self._on_interject_done)

    def _on_interject_done(self, text: str | None) -> None:
        if text is None:
            return
        replacing = self._pending_directive is not None
        self._pending_directive = text
        preview = text if len(text) <= 60 else text[:60] + "…"
        msg = "Directive replaced" if replacing else "Directive queued"
        self.notify(f'{msg}: "{preview}"')

    def _consume_directive(self) -> str | None:
        d, self._pending_directive = self._pending_directive, None
        return d
