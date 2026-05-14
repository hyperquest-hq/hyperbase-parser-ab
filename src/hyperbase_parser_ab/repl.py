"""REPL integration for the AlphaBeta parser.

Adds a pre-result hook to the Hyperbase REPL that prints the spaCy
dependency parse tree for the current sentence. Imported lazily from
:meth:`AlphaBetaParser.install_repl` so that this module's only purpose
is keeping REPL-rendering code out of the parser core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hyperbase.parsers.repl_api import PreResultHook, ReplContext
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from spacy.tokens import Token

from hyperbase_parser_ab.trace import ParseTrace, RuleIteration, SubstitutionRound

if TYPE_CHECKING:
    from hyperbase_parser_ab.parser import AlphaBetaParser


def _build_dependency_tree(
    token: Token,
    visited: set[Token] | None = None,
) -> Tree | None:
    """Build a Rich tree representation of a spaCy dependency parse."""
    if visited is None:
        visited = set()

    if token in visited:
        return None
    visited.add(token)

    label = Text()
    label.append(token.text, style="bold white")
    label.append(" [", style="dim")
    label.append(f"dep_={token.dep_}", style="cyan")
    label.append(", ", style="dim")
    label.append(f"tag_={token.tag_}", style="green")
    label.append(", ", style="dim")
    label.append(f"pos_={token.pos_}", style="yellow")
    label.append("]", style="dim")

    tree = Tree(label)

    for child in token.children:
        child_tree = _build_dependency_tree(child, visited)
        if child_tree:
            tree.add(child_tree)

    return tree


def _make_pre_result_hook(parser: AlphaBetaParser) -> PreResultHook:
    """Return a pre-result hook that prints the dep tree of the current
    parse's spaCy span."""
    del parser  # unused, kept for symmetry with other hook factories

    def hook(ctx: ReplContext) -> None:
        if ctx.result is None:
            return
        sent = ctx.result.extra.get("spacy_sent")
        if sent is None:
            return
        dep_tree = _build_dependency_tree(sent.root)
        if dep_tree is None:
            return
        console: Console = ctx.session.console
        console.print()
        console.print(
            Panel(
                dep_tree,
                title="[bold cyan]Dependency Parse Tree[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

    return hook


def _atoms_panel(trace: ParseTrace) -> Panel:
    table = Table(box=box.SIMPLE, padding=(0, 1), show_header=True)
    table.add_column("idx", style="dim", justify="right")
    table.add_column("token", style="bold white")
    table.add_column("predicted", style="cyan")
    table.add_column("refined", style="yellow")
    table.add_column("atom", style="green")
    table.add_column("dropped", style="red", justify="center")
    table.add_column("uncertain", style="magenta", justify="center")
    table.add_column("top-3", style="dim")

    for at in trace.atoms:
        if at.top_candidates:
            top3 = ", ".join(f"{lbl}:{prob:.5f}" for lbl, prob in at.top_candidates[:3])
        else:
            top3 = "—"
        table.add_row(
            str(at.token_idx),
            at.token_text,
            at.predicted_type,
            at.refined_type,
            at.final_atom or "—",
            "✓" if at.dropped else "",
            "✓" if at.is_uncertain else "",
            top3,
        )

    return Panel(
        table,
        title="[bold cyan]Atoms[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    )


def _iteration_panel(it: RuleIteration) -> Panel:
    seq_text = Text()
    seq_text.append("sequence:\n", style="bold")
    for i, item in enumerate(it.sequence_repr):
        seq_text.append(f"  [{i}] ", style="dim")
        seq_text.append(f"{item}\n")

    candidates_table = Table(box=box.SIMPLE, padding=(0, 1), show_header=True)
    candidates_table.add_column("rule", style="white")
    candidates_table.add_column("pos", style="dim", justify="right")
    candidates_table.add_column("badness", style="dim", justify="right")
    candidates_table.add_column("distortion", style="dim", justify="right")
    candidates_table.add_column("score", style="dim", justify="right")
    candidates_table.add_column("new edge", style="white")

    if it.candidates:
        for cand in it.candidates:
            style = "bold green" if cand.is_winner else "white"
            marker = "★ " if cand.is_winner else "  "
            candidates_table.add_row(
                Text(f"{marker}{cand.rule_repr}", style=style),
                Text(str(cand.pos), style=style),
                Text(str(cand.badness), style=style),
                Text(str(cand.distortion), style=style),
                Text(str(cand.score), style=style),
                Text(cand.new_edge_repr, style=style),
            )
    else:
        candidates_table.add_row(
            Text("(no candidates)", style="dim italic"), "", "", "", "", ""
        )

    body: list[object] = [seq_text, candidates_table]

    if it.dominated:
        dominated_table = Table(box=box.SIMPLE, padding=(0, 1), show_header=True)
        dominated_table.add_column("rule", style="dim")
        dominated_table.add_column("pos", style="dim", justify="right")
        dominated_table.add_column("badness", style="dim", justify="right")
        dominated_table.add_column("distortion", style="dim", justify="right")
        dominated_table.add_column("score", style="dim", justify="right")
        dominated_table.add_column("new edge", style="dim")
        for cand in it.dominated:
            dominated_table.add_row(
                Text(f"  {cand.rule_repr}", style="dim red"),
                Text(str(cand.pos), style="dim red"),
                Text(str(cand.badness), style="dim red"),
                Text(str(cand.distortion), style="dim red"),
                Text(str(cand.score), style="dim red"),
                Text(cand.new_edge_repr, style="dim red"),
            )
        body.append(Text("dominated:", style="dim red bold"))
        body.append(dominated_table)

    border = "yellow" if it.fallback_used else "magenta"
    title = (
        f"[bold magenta]Iteration {it.iteration}[/bold magenta]"
        if not it.fallback_used
        else f"[bold yellow]Iteration {it.iteration} (fallback)[/bold yellow]"
    )
    return Panel(Group(*body), title=title, border_style=border, box=box.ROUNDED)


def _final_badness_panel(trace: ParseTrace) -> Panel | None:
    if not trace.final_badness:
        return None
    table = Table(box=box.SIMPLE, padding=(0, 1), show_header=True)
    table.add_column("location", style="white")
    table.add_column("severity", style="dim", justify="right")
    table.add_column("type", style="yellow")
    table.add_column("message", style="white")
    for location, issues in trace.final_badness.items():
        for err_type, message, severity in issues:
            table.add_row(location, str(severity), err_type, message)
    return Panel(
        table,
        title="[bold red]Final badness[/bold red]",
        border_style="red",
        box=box.ROUNDED,
    )


def _substitution_round_panel(round_: SubstitutionRound) -> Panel:
    seed_text = Text()
    seed_text.append("starting cost: ", style="bold")
    seed_text.append(
        f"badness={round_.seed_badness} "
        f"distortion={round_.seed_distortion} "
        f"score={round_.seed_score}\n",
        style="dim",
    )

    table = Table(box=box.SIMPLE, padding=(0, 1), show_header=True)
    table.add_column("", style="bold green", no_wrap=True)
    table.add_column("#", style="dim", justify="right", no_wrap=True)
    table.add_column("token", style="bold white")
    table.add_column("from", style="cyan")
    table.add_column("to", style="yellow")
    table.add_column("badness", style="dim", justify="right")
    table.add_column("distortion", style="dim", justify="right")
    table.add_column("score", style="dim", justify="right")
    table.add_column("parse", style="white")

    if not round_.trials:
        table.add_row(
            "", "", Text("(no trials)", style="dim italic"), "", "", "", "", "", ""
        )
    else:
        for trial in round_.trials:
            style = "bold green" if trial.is_winner else "white"
            marker = "★" if trial.is_winner else ""
            table.add_row(
                Text(marker, style="bold green"),
                Text(str(trial.number), style=style),
                Text(trial.token_text, style=style),
                Text(trial.label_from, style=style),
                Text(trial.label_to, style=style),
                Text(str(trial.badness), style=style),
                Text(str(trial.distortion), style=style),
                Text(str(trial.score), style=style),
                Text(trial.edge_repr, style=style),
            )

    border = "magenta" if round_.improved else "yellow"
    suffix = " (no improvement)" if not round_.improved else ""
    title = "[bold magenta]Substitution Round "
    f"{round_.round_idx}{suffix}[/bold magenta]"
    return Panel(
        Group(seed_text, table), title=title, border_style=border, box=box.ROUNDED
    )


def _totals_panel(trace: ParseTrace) -> Panel:
    table = Table(box=box.SIMPLE, padding=(0, 1), show_header=False)
    table.add_column("metric", style="bold white")
    table.add_column("value", style="cyan", justify="right")
    table.add_row("final badness", str(trace.total_badness))
    table.add_row("final distortion", str(trace.total_distortion))
    return Panel(
        table,
        title="[bold cyan]Totals[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    )


def _passes_panel(trace: ParseTrace) -> Panel | None:
    if not trace.passes:
        return None
    table = Table(box=box.SIMPLE, padding=(0, 1), show_header=True)
    table.add_column("pass", style="cyan", justify="right")
    table.add_column("stranded atoms", style="white")
    for idx, strands in enumerate(trace.passes, start=1):
        atoms_repr = ", ".join(strands) if strands else "(none)"
        table.add_row(str(idx), atoms_repr)
    title = f"[bold cyan]Parse passes ({len(trace.passes)})[/bold cyan]"
    return Panel(table, title=title, border_style="cyan", box=box.ROUNDED)


def _post_processing_panel(trace: ParseTrace) -> Panel | None:
    if not trace.post_processing:
        return None
    table = Table(box=box.SIMPLE, padding=(0, 1), show_header=True)
    table.add_column("stage", style="cyan")
    table.add_column("edge", style="white")
    for stage, edge_repr in trace.post_processing:
        table.add_row(stage, edge_repr)
    return Panel(
        table,
        title="[bold cyan]Post-processing[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    )


def _make_report_hook(parser: AlphaBetaParser) -> PreResultHook:
    """Return a pre-result hook that prints the structured parse trace
    when the ``report`` REPL setting is enabled."""
    del parser

    def hook(ctx: ReplContext) -> None:
        if ctx.result is None:
            return
        trace = ctx.result.extra.get("parse_trace")
        if not isinstance(trace, ParseTrace):
            return
        _render_report(ctx.session.console, trace)

    return hook


def _render_report(console: Console, trace: ParseTrace) -> None:
    """Print the full structured-parse report. Used both by the
    pre-result hook (when ``report`` is on for a normal parse) and by
    the ``/sub`` command when replaying a substitution trial."""
    console.print()
    console.print(_atoms_panel(trace))

    for it in trace.iterations:
        console.print(_iteration_panel(it))

    for round_ in trace.substitution_rounds:
        console.print(_substitution_round_panel(round_))

    passes_panel = _passes_panel(trace)
    if passes_panel is not None:
        console.print(passes_panel)

    post_panel = _post_processing_panel(trace)
    if post_panel is not None:
        console.print(post_panel)

    final_badness_panel = _final_badness_panel(trace)
    if final_badness_panel is not None:
        console.print(final_badness_panel)

    console.print(_totals_panel(trace))


def install(parser: AlphaBetaParser, session: object) -> None:
    """Register AlphaBeta-specific REPL behavior on *session*.

    Only ``report`` is registered here. ``post_processing`` is already
    in ``accepted_params``, which means the REPL exposes it to ``/set``
    and ``/settings`` automatically. Registering it again as a plugin
    setting would put it in ``_extra_settings``, and the REPL's
    ``_reset_plugin_state`` (run before each parser rebuild) pops
    everything in ``_extra_settings`` from the live settings dict — so
    the user's ``/set`` value would be wiped before
    ``_build_parser_kwargs`` could read it back into the new parser
    instance.
    """
    session.register_setting(  # type: ignore[attr-defined]
        "report",
        default=False,
        type_=bool,
        description="Show detailed parse trace (atoms, rules, transformations).",
    )
    session.register_pre_result_hook(_make_pre_result_hook(parser))  # type: ignore[attr-defined]
    session.register_pre_result_hook(_make_report_hook(parser))  # type: ignore[attr-defined]
    session.register_command(  # type: ignore[attr-defined]
        "sub",
        "Re-parse last sentence with substitutions from trial #N (see /sub <N>).",
        _make_sub_command(parser, session),
    )
    session.register_command(  # type: ignore[attr-defined]
        "dpt",
        "Re-parse last sentence and dump per-edge distortion analysis.",
        _make_dpt_command(parser, session),
    )


def _make_sub_command(
    parser: AlphaBetaParser, session: object
) -> callable[[list[str]], bool]:
    def cmd_sub(args: list[str]) -> bool:
        console: Console = session.console  # type: ignore[attr-defined]
        if not args:
            console.print(
                "[red]Error:[/red] /sub requires a trial number ([cyan]/sub <N>[/cyan])"
            )
            return False
        try:
            number = int(args[0])
        except ValueError:
            console.print(f"[red]Error:[/red] invalid number: [cyan]{args[0]}[/cyan]")
            return False
        state = parser._last_sub_climb
        if state is None:
            console.print(
                "[yellow]No substitution trials cached. "
                "Parse a sentence first.[/yellow]"
            )
            return False
        trial_subs = state["trial_subs"]
        if number not in trial_subs:
            nums = sorted(trial_subs)
            available = f"{nums[0]}..{nums[-1]}" if nums else "(none)"
            console.print(
                f"[red]Error:[/red] trial #{number} not found "
                f"(available: [cyan]{available}[/cyan])"
            )
            return False
        forced = trial_subs[number]
        sub_repr = (
            ", ".join(f"tok{idx}→{lbl}" for idx, lbl in sorted(forced.items()))
            if forced
            else "(none)"
        )
        console.print(
            f"[dim]Replaying trial #{number} with substitutions:[/dim] "
            f"[cyan]{sub_repr}[/cyan]"
        )
        results = parser.parse_sentence_with_substitutions(state["sentence"], forced)
        if not results:
            console.print("[yellow]No parse produced.[/yellow]")
            return False
        report_on: bool = bool(session.settings.get("report", False))  # type: ignore[attr-defined]
        for r in results:
            console.print(str(r.edge))
            if report_on:
                trace = r.extra.get("parse_trace")
                if isinstance(trace, ParseTrace):
                    _render_report(console, trace)
        return False

    return cmd_sub


def _make_dpt_command(
    parser: AlphaBetaParser, session: object
) -> callable[[list[str]], bool]:
    def cmd_dpt(args: list[str]) -> bool:
        console: Console = session.console  # type: ignore[attr-defined]
        sentence = parser._last_parsed_sentence
        if sentence is None:
            console.print("[yellow]No cached parse. Parse a sentence first.[/yellow]")
            return False
        # Re-parse so parser state (orig_atom, _dpt_directed, token2atom)
        # is freshly populated for the diagnostic walk.
        results = parser.parse_sentence(sentence)
        if not results:
            console.print("[yellow]No parse produced.[/yellow]")
            return False
        # `parse_sentence` resets and rebuilds state per sub-sentence;
        # for multi-sentence input only the last sub-sentence's state
        # survives. Diagnose the matching result.
        target = results[-1]
        console.print()
        console.print(
            f"[bold cyan]DPT directed edges ({len(parser._dpt_directed)}):[/bold cyan]"
        )
        for ua_parent, ua_child in sorted(
            parser._dpt_directed, key=lambda p: (str(p[0]), str(p[1]))
        ):
            console.print(f"  {ua_parent} → {ua_child}")
        console.print()
        console.print(f"[bold cyan]Final edge:[/bold cyan] {target.edge}")
        console.print()
        console.print("[bold cyan]Per-edge distortion analysis:[/bold cyan]")
        for line in parser.diagnose_distortion(target.edge):
            console.print(line)
        return False

    return cmd_dpt
