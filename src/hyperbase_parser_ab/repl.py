"""REPL integration for the AlphaBeta parser.

Adds a pre-result hook to the Hyperbase REPL that prints the spaCy
dependency parse tree for the current sentence. Imported lazily from
:meth:`AlphaBetaParser.install_repl` so that this module's only purpose
is keeping REPL-rendering code out of the parser core.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from hyperbase.parsers.repl_api import PreResultHook, ReplContext
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from spacy.tokens import Token

from hyperbase_parser_ab.trace import ParseTrace, RuleIteration

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
    candidates_table.add_column("score", style="dim", justify="right")
    candidates_table.add_column("new edge", style="white")

    if it.candidates:
        for cand in it.candidates:
            style = "bold green" if cand.is_winner else "white"
            marker = "★ " if cand.is_winner else "  "
            candidates_table.add_row(
                Text(f"{marker}{cand.rule_repr}", style=style),
                Text(str(cand.pos), style=style),
                Text(str(cand.score), style=style),
                Text(cand.new_edge_repr, style=style),
            )
    else:
        candidates_table.add_row(
            Text("(no candidates)", style="dim italic"), "", "", ""
        )

    body: list[object] = [seq_text, candidates_table]

    if it.rejections:
        grouped: dict[int, list[int]] = defaultdict(list)
        for rule_idx, pos in it.rejections:
            grouped[rule_idx].append(pos)
        parts = [
            f"rule#{r} @ pos {','.join(str(p) for p in sorted(positions))}"
            for r, positions in sorted(grouped.items())
        ]
        rejected_text = Text()
        rejected_text.append("Rejected: ", style="dim red bold")
        rejected_text.append("; ".join(parts), style="dim red")
        body.append(rejected_text)

    border = "yellow" if it.fallback_used else "magenta"
    title = (
        f"[bold magenta]Iteration {it.iteration}[/bold magenta]"
        if not it.fallback_used
        else f"[bold yellow]Iteration {it.iteration} (fallback)[/bold yellow]"
    )
    return Panel(Group(*body), title=title, border_style=border, box=box.ROUNDED)


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

        console: Console = ctx.session.console
        console.print()
        console.print(_atoms_panel(trace))

        for it in trace.iterations:
            console.print(_iteration_panel(it))

        post_panel = _post_processing_panel(trace)
        if post_panel is not None:
            console.print(post_panel)

    return hook


def install(parser: AlphaBetaParser, session: object) -> None:
    """Register AlphaBeta-specific REPL behavior on *session*."""
    session.register_setting(  # type: ignore[attr-defined]
        "report",
        default=False,
        type_=bool,
        description="Show detailed parse trace (atoms, rules, transformations).",
    )
    session.register_pre_result_hook(_make_pre_result_hook(parser))  # type: ignore[attr-defined]
    session.register_pre_result_hook(_make_report_hook(parser))  # type: ignore[attr-defined]
