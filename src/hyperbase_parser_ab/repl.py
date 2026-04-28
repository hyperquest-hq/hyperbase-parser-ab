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
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from spacy.tokens import Token

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
    label.append(f"tag_={token.pos_}", style="yellow")
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


def install(parser: AlphaBetaParser, session: object) -> None:
    """Register AlphaBeta-specific REPL behavior on *session*."""
    session.register_pre_result_hook(_make_pre_result_hook(parser))  # type: ignore[attr-defined]
