"""`/genparse` REPL command — interactive corpus generation.

The user feeds a .jsonl parse-results file. For each input text, the
AlphaBeta parser runs once in automatic mode and the result is shown.
The user can accept the automatic parse as-is, re-run the parse in
manual mode (picking the winning rule candidate at every reduction
step), skip the sentence, pause, or quit.

Every accepted parse is appended to the output .jsonl as one record per
sentence (a spaCy doc may split an input into several sentences). Each
record carries the full per-token spaCy features plus the top-3
atomizer labels, and every candidate set the parser considered with the
correct choice flagged. The aim is enough information to train a
candidate classifier that replaces the current
``(badness, distortion, score)`` heuristic.

State persistence:
- The last input/output paths and a deterministic shuffle seed (one
  per input path) are kept in ``~/.hyperbase_repl_settings.json`` under
  the ``genparse`` key. ``/genparse`` with no args reuses them.
- Resume is by counting the lines already in the output file and
  skipping that many entries in the shuffled order. Because the
  shuffle is seeded by a hash of the absolute input path, re-runs
  produce the same order — safe to resume across sessions.

Registered by :func:`hyperbase_parser_ab.repl.install`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperbase.cli.repl import save_settings
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from spacy.tokens import Span, Token

from hyperbase_parser_ab.trace import (
    ManualCandidate,
    ParseTrace,
    RuleIteration,
)

if TYPE_CHECKING:
    from hyperbase.parsers.parse_result import ParseResult

    from hyperbase_parser_ab.parser import AlphaBetaParser


# Settings dict layout under settings["genparse"]:
#   {
#     "input": "/abs/path/to/input.jsonl",
#     "output": "/abs/path/to/output.jsonl",
#     "seeds": { "<input_path>": <int seed> },
#   }
_GENPARSE_SETTING: str = "genparse"


def _serialize_token(token: Token, sent_start: int) -> dict[str, Any]:
    """spaCy Token → JSON-safe dict with enough info to reconstruct a DPT.

    `head_i` is sentence-relative (the token's head index minus
    `sent_start`), so a consumer can rebuild the tree without knowing
    the surrounding doc."""
    head_i: int = token.head.i - sent_start
    return {
        "i": token.i - sent_start,
        "text": token.text,
        "lemma_": token.lemma_,
        "pos_": token.pos_,
        "tag_": token.tag_,
        "dep_": token.dep_,
        "head_i": head_i,
        "morph": str(token.morph),
        "is_alpha": bool(token.is_alpha),
        "is_punct": bool(token.is_punct),
    }


def _serialize_tokens(sent: Span, trace: ParseTrace | None) -> list[dict[str, Any]]:
    """Per-sentence token list with top-3 atomizer labels merged in.

    The atomizer's per-token (label, prob) lists live on the
    ParseTrace via :class:`AtomTrace` and are indexed by sentence
    position. When the trace is missing or has no atoms, the
    `atomizer_top3` field is an empty list (still valid as a feature
    column — the classifier will just learn nothing from it)."""
    top_by_idx: dict[int, list[tuple[str, float]]] = {}
    if trace is not None:
        for at in trace.atoms:
            if at.top_candidates:
                top_by_idx[at.token_idx] = list(at.top_candidates[:3])
    tokens: list[dict[str, Any]] = []
    for sent_idx, tok in enumerate(sent):
        d = _serialize_token(tok, sent.start)
        d["atomizer_top3"] = [
            [lbl, float(prob)] for lbl, prob in top_by_idx.get(sent_idx, [])
        ]
        tokens.append(d)
    return tokens


def _serialize_iterations(trace: ParseTrace) -> list[dict[str, Any]]:
    """ParseTrace.iterations → list of decision-step dicts.

    Each step exposes the full surviving candidate set (post
    dominance-filter, pre winner-pick) so a classifier sees the same
    options the parser did. `is_correct` mirrors the trace's
    `is_winner`, which in manual mode reflects the user's pick and in
    auto mode reflects the automatic comparator's pick."""
    out: list[dict[str, Any]] = []
    for it in trace.iterations:
        cand_dicts: list[dict[str, Any]] = []
        for c in it.candidates:
            cand_dicts.append(
                {
                    "rule_index": c.rule_index,
                    "rule_repr": c.rule_repr,
                    "pos": c.pos,
                    "badness": c.badness,
                    "distortion": c.distortion,
                    "score": c.score,
                    "no_dangling": c.no_dangling,
                    "indices": c.indices,
                    "new_edge_repr": c.new_edge_repr,
                    "is_correct": c.is_winner,
                }
            )
        out.append(
            {
                "iteration": it.iteration,
                "sequence_repr": list(it.sequence_repr),
                "fallback_used": it.fallback_used,
                "candidates": cand_dicts,
            }
        )
    return out


def _build_record(
    parse: ParseResult,
    language: str,
    mode: str,
) -> dict[str, Any]:
    """Assemble the per-sentence output record."""
    sent: Span | None = parse.extra.get("spacy_sent")
    trace: ParseTrace | None = parse.extra.get("parse_trace")
    tokens: list[dict[str, Any]] = (
        _serialize_tokens(sent, trace) if sent is not None else []
    )
    decisions: list[dict[str, Any]] = (
        _serialize_iterations(trace) if trace is not None else []
    )
    return {
        "text": parse.text,
        "language": language,
        "mode": mode,
        "edge": str(parse.edge),
        "tokens": tokens,
        "decisions": decisions,
    }


def _hash_seed(path: Path) -> int:
    """Stable shuffle seed derived from the absolute input path."""
    h = hashlib.sha256(str(path.resolve()).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def _load_input_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = d.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
    return texts


def _count_output_records(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _render_candidates_table(cands: list[ManualCandidate], default_idx: int) -> Table:
    """Two-column-ish table for manual candidate selection."""
    table = Table(box=box.SIMPLE, padding=(0, 1), show_header=True)
    table.add_column("", style="dim", justify="right")
    table.add_column("#", style="bold cyan", justify="right")
    table.add_column("rule", style="white")
    table.add_column("pos", style="dim", justify="right")
    table.add_column("bad", style="dim", justify="right")
    table.add_column("dist", style="dim", justify="right")
    table.add_column("score", style="dim", justify="right")
    table.add_column("nd", style="dim", justify="center")
    table.add_column("sw", style="dim", justify="center")
    table.add_column("new edge", style="white")
    for i, c in enumerate(cands):
        marker = "→" if i == default_idx else ""
        style = "bold green" if i == default_idx else None
        cells = [
            marker,
            str(i),
            c.rule_repr,
            str(c.pos),
            str(c.badness),
            str(c.distortion),
            str(c.score),
            "✓" if c.no_dangling else "",
            "✓" if c.is_sliding_window else "",
            c.new_edge_repr,
        ]
        if style is not None:
            table.add_row(*[Text(x, style=style) for x in cells])
        else:
            table.add_row(*cells)
    return table


def _make_manual_picker(session: object) -> Callable[[list[ManualCandidate], int], int]:
    """Returns the ManualPickFn used by parse_sentence in manual mode.

    Raises ``_GenparseAbortError`` to unwind the parse when the user wants to
    quit/cancel mid-sentence (the parser catches arbitrary exceptions
    and aborts the parse cleanly, so we use a sentinel exception
    rather than the int return value)."""
    console: Console = session.console  # type: ignore[attr-defined]
    prompt_session = session.session  # type: ignore[attr-defined]
    step_idx: dict[str, int] = {"i": 0}

    def picker(cands: list[ManualCandidate], default_idx: int) -> int:
        step = step_idx["i"]
        step_idx["i"] = step + 1
        console.print()
        console.print(
            Panel(
                _render_candidates_table(cands, default_idx),
                title=(
                    f"[bold magenta]Decision #{step}[/bold magenta] "
                    f"[dim]({len(cands)} candidates)[/dim]"
                ),
                border_style="magenta",
                box=box.ROUNDED,
            )
        )
        while True:
            try:
                ans = prompt_session.prompt(
                    f"pick [0..{len(cands) - 1}] (Enter={default_idx}) > "
                ).strip()
            except (KeyboardInterrupt, EOFError):
                raise _GenparseAbortError() from None
            if ans == "":
                return default_idx
            try:
                idx = int(ans)
            except ValueError:
                console.print("[red]not a number[/red]")
                continue
            if 0 <= idx < len(cands):
                return idx
            console.print(f"[red]out of range (0..{len(cands) - 1})[/red]")

    return picker


class _GenparseAbortError(Exception):
    """Raised by the manual picker to abort the current sentence's parse."""


def _render_text_header(
    console: Console,
    text: str,
    cur: int,
    total: int,
    accepted: int,
) -> None:
    title = (
        f"[bold yellow]Input {cur + 1}/{total}[/bold yellow] "
        f"[dim](accepted: {accepted})[/dim]"
    )
    console.print()
    console.print(
        Panel(
            Text(text, style="italic"),
            title=title,
            border_style="yellow",
            box=box.ROUNDED,
        )
    )


def _render_diagnostics(session: object, parses: list[ParseResult]) -> None:
    """Per-sentence dependency parse tree and atomizer top-3 panels.

    Reuses the same renderers as the REPL `/set report=on` panels so
    the user sees the exact same view that report mode produces. Both
    panels are no-ops when their source data is missing (parse failed
    before producing a spaCy span / before populating atom traces)."""
    # Lazy import: genparse and repl live in the same package, and
    # repl.install() imports genparse to register /genparse. Importing
    # the renderers at module top level would still work (repl
    # finishes loading before install() runs), but the lazy import
    # keeps the dependency direction one-way for any future reader.
    from hyperbase_parser_ab.repl import _atoms_panel, _build_dependency_tree

    console: Console = session.console  # type: ignore[attr-defined]
    for p in parses:
        sent = p.extra.get("spacy_sent")
        if sent is not None:
            dep_tree = _build_dependency_tree(sent.root)
            if dep_tree is not None:
                console.print()
                console.print(
                    Panel(
                        dep_tree,
                        title="[bold cyan]Dependency Parse Tree[/bold cyan]",
                        border_style="cyan",
                        box=box.ROUNDED,
                    )
                )
        trace = p.extra.get("parse_trace")
        if isinstance(trace, ParseTrace) and trace.atoms:
            console.print(_atoms_panel(trace))


def _render_parse_panels(session: object, parses: list[ParseResult]) -> None:
    console: Console = session.console  # type: ignore[attr-defined]
    formatter = session.formatter  # type: ignore[attr-defined]
    for i, p in enumerate(parses):
        title = (
            "[yellow]Parse[/yellow]"
            if len(parses) == 1
            else f"[yellow]Parse {i + 1}/{len(parses)}[/yellow]"
        )
        if len(parses) > 1:
            console.print(Text(p.text, style="dim italic"))
        if p.edge is None:
            body: Any = Text("FAILED", style="bold red")
            border = "red"
        else:
            body = formatter.format(p.edge)
            border = "green"
        console.print(Panel(body, title=title, border_style=border, box=box.ROUNDED))


def _fmt_hours_rate(accepted: int, active_seconds: float) -> str:
    if active_seconds <= 0 or accepted == 0:
        return "—/h"
    rate = accepted / (active_seconds / 3600.0)
    return f"{rate:.1f}/h"


def _render_stats(
    console: Console,
    accepted: int,
    target_total: int,
    edge_sizes: list[int],
    active_seconds: float,
    paused: bool,
) -> None:
    table = Table(box=box.SIMPLE, padding=(0, 1), show_header=False)
    table.add_column("", style="cyan")
    table.add_column("", style="green", justify="right")
    table.add_row("accepted", f"{accepted}/{target_total}")
    if edge_sizes:
        n = len(edge_sizes)
        s_min = min(edge_sizes)
        s_max = max(edge_sizes)
        s_mean = sum(edge_sizes) / n
        table.add_row("edge atoms (min/mean/max)", f"{s_min} / {s_mean:.1f} / {s_max}")
    else:
        table.add_row("edge atoms (min/mean/max)", "—")
    table.add_row("active time", f"{active_seconds / 60.0:.1f} min")
    table.add_row("rate", _fmt_hours_rate(accepted, active_seconds))
    title_suffix = " [yellow](paused)[/yellow]" if paused else ""
    console.print(
        Panel(
            table,
            title=f"[bold blue]/genparse stats[/bold blue]{title_suffix}",
            border_style="blue",
            box=box.ROUNDED,
        )
    )


def _resolve_paths(
    args: list[str], saved: dict[str, Any]
) -> tuple[Path | None, Path | None, str | None]:
    """Return (input_path, output_path, error_msg)."""
    input_str: str | None
    output_str: str | None
    if len(args) >= 2:
        input_str = args[0]
        output_str = args[1]
    elif len(args) == 1:
        return (
            None,
            None,
            ("Provide both input and output paths, or none to reuse the last ones."),
        )
    else:
        input_str = saved.get("input")
        output_str = saved.get("output")
        if not input_str or not output_str:
            return (
                None,
                None,
                ("No saved paths. Usage: /genparse <input.jsonl> <output.jsonl>"),
            )
    input_path = Path(input_str).expanduser()
    output_path = Path(output_str).expanduser()
    if not input_path.is_file():
        return None, None, f"input file not found: {input_path}"
    return input_path, output_path, None


def _make_genparse_command(
    parser: AlphaBetaParser, session: object
) -> Callable[[list[str]], bool]:
    """Build the /genparse REPL command handler bound to *parser* / *session*."""

    def cmd_genparse(args: list[str]) -> bool:
        console: Console = session.console  # type: ignore[attr-defined]
        prompt_session = session.session  # type: ignore[attr-defined]
        settings: dict[str, Any] = session.settings  # type: ignore[attr-defined]
        saved: dict[str, Any] = dict(settings.get(_GENPARSE_SETTING) or {})

        input_path, output_path, err = _resolve_paths(args, saved)
        if err is not None:
            console.print(f"[red]Error:[/red] {err}")
            return False
        assert input_path is not None
        assert output_path is not None

        # Persist paths and shuffle seed (one per input). Reusing the
        # same seed for the same input is what makes resume work — the
        # shuffled order must be reproducible across sessions.
        seeds_map: dict[str, int] = dict(saved.get("seeds") or {})
        seed_key: str = str(input_path.resolve())
        if seed_key not in seeds_map:
            seeds_map[seed_key] = _hash_seed(input_path)
        seed: int = seeds_map[seed_key]
        settings[_GENPARSE_SETTING] = {
            "input": str(input_path),
            "output": str(output_path),
            "seeds": seeds_map,
        }
        save_settings(settings)

        try:
            texts = _load_input_texts(input_path)
        except OSError as e:
            console.print(f"[red]Error:[/red] failed to read {input_path}: {e}")
            return False
        if not texts:
            console.print(
                f"[yellow]No texts found in[/yellow] [cyan]{input_path}[/cyan]"
            )
            return False

        indices = list(range(len(texts)))
        random.Random(seed).shuffle(indices)

        already = _count_output_records(output_path)
        if already >= len(indices):
            console.print(
                f"[green]Already done:[/green] {already} record(s) in "
                f"[cyan]{output_path}[/cyan] (input has {len(texts)} text(s))"
            )
            return False
        if already > 0:
            console.print(
                f"[dim]Resuming from[/dim] [cyan]{output_path}[/cyan] "
                f"[dim]({already} already accepted)[/dim]"
            )

        # Make sure the output directory exists so the first append
        # doesn't ENOENT halfway through a session.
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            console.print(f"[red]Error:[/red] cannot create parent dir: {e}")
            return False

        language: str = str(settings.get("language") or "en")
        # `position` walks the shuffled order; `accepted` tracks how
        # many records are in the output file. Within a session, skip
        # advances `position` but not `accepted` so the user moves on
        # to the next text. Across sessions, resume keys off `accepted`
        # only (= output file line count), so any sentences skipped in
        # a prior session will come back — the user can re-skip them
        # or accept this time.
        accepted: int = already
        position: int = already
        edge_sizes: list[int] = []
        active_seconds: float = 0.0
        seg_start: float = time.perf_counter()
        target_total: int = len(indices)

        # Don't pollute the REPL command history with single-letter
        # answers ("a", "m", "s", "q", numeric picks).
        history = session.history  # type: ignore[attr-defined]
        history.paused = True

        console.print(
            f"[bold]/genparse[/bold] [dim]input=[/dim][cyan]{input_path}[/cyan] "
            f"[dim]output=[/dim][cyan]{output_path}[/cyan]"
        )
        console.print(
            f"[dim]shuffle seed: {seed} | {target_total - already} text(s) to go[/dim]"
        )

        def _active_seconds_now() -> float:
            return active_seconds + (time.perf_counter() - seg_start)

        try:
            while position < target_total:
                cur = position
                text = texts[indices[cur]]
                _render_text_header(console, text, cur, target_total, accepted)

                # Run automatic parse first — the user almost always
                # accepts it, so showing it up front and prompting once
                # is the fast path. `force_trace=True` populates the
                # per-iteration candidate trace even though no manual
                # picker is installed, so an accepted auto parse can
                # serialize the same decision-step records as a manual
                # one (each step's winner is the automatic comparator's
                # pick).
                try:
                    auto_parses = parser.parse_sentence(text, force_trace=True)
                except Exception as e:
                    console.print(f"[red]Auto parse crashed:[/red] {e}")
                    auto_parses = []
                _render_diagnostics(session, auto_parses)
                _render_parse_panels(session, auto_parses)

                while True:
                    try:
                        ans = (
                            prompt_session.prompt(
                                "[a]ccept  [m]anual  [s]kip  [p]ause  [t]stats  "
                                "[q]uit > "
                            )
                            .strip()
                            .lower()
                        )
                    except (KeyboardInterrupt, EOFError):
                        ans = "q"

                    if ans in ("", "a", "accept"):
                        if not auto_parses:
                            console.print(
                                "[yellow]No parse produced. Try /m to enter manual "
                                "mode or /s to skip.[/yellow]"
                            )
                            continue
                        _append_records(output_path, auto_parses, language, "auto")
                        for p in auto_parses:
                            if p.edge is not None:
                                edge_sizes.append(len(p.edge.all_atoms()))
                        accepted += 1
                        position += 1
                        break

                    if ans in ("m", "manual"):
                        picker = _make_manual_picker(session)
                        try:
                            manual_parses = parser.parse_sentence(
                                text, manual_pick=picker
                            )
                        except _GenparseAbortError:
                            console.print("[dim](manual parse aborted)[/dim]")
                            continue
                        except Exception as e:
                            console.print(f"[red]Manual parse crashed:[/red] {e}")
                            continue
                        _render_parse_panels(session, manual_parses)
                        try:
                            confirm = (
                                prompt_session.prompt(
                                    "Save this manual parse? [Y/n] > "
                                )
                                .strip()
                                .lower()
                            )
                        except (KeyboardInterrupt, EOFError):
                            confirm = "n"
                        if confirm in ("", "y", "yes"):
                            _append_records(
                                output_path, manual_parses, language, "manual"
                            )
                            for p in manual_parses:
                                if p.edge is not None:
                                    edge_sizes.append(len(p.edge.all_atoms()))
                            accepted += 1
                            position += 1
                            break
                        continue

                    if ans in ("s", "skip"):
                        # In-session advance only — the skipped record
                        # is not in the output file, so a future
                        # /genparse run on the same input/output pair
                        # will see it again at the same shuffled slot
                        # (resume only counts written records).
                        position += 1
                        break

                    if ans in ("t", "stats"):
                        _render_stats(
                            console,
                            accepted,
                            target_total,
                            edge_sizes,
                            _active_seconds_now(),
                            paused=False,
                        )
                        continue

                    if ans in ("p", "pause"):
                        active_seconds += time.perf_counter() - seg_start
                        _render_stats(
                            console,
                            accepted,
                            target_total,
                            edge_sizes,
                            active_seconds,
                            paused=True,
                        )
                        with contextlib.suppress(KeyboardInterrupt, EOFError):
                            prompt_session.prompt("[paused] press Enter to resume > ")
                        seg_start = time.perf_counter()
                        continue

                    if ans in ("q", "quit", "exit"):
                        raise _GenparseQuitError()

                    console.print("[red]unknown command[/red] [dim](a/m/s/p/t/q)[/dim]")

            console.print(f"[green]✓ Done.[/green] {accepted}/{target_total} accepted.")
        except _GenparseQuitError:
            console.print("[yellow]Quitting /genparse.[/yellow]")
        finally:
            active_seconds += time.perf_counter() - seg_start
            history.paused = False
            _render_stats(
                console,
                accepted,
                target_total,
                edge_sizes,
                active_seconds,
                paused=False,
            )
        return False

    return cmd_genparse


class _GenparseQuitError(Exception):
    """Raised internally to exit the /genparse outer loop on user 'q'."""


def _append_records(
    output_path: Path,
    parses: list[ParseResult],
    language: str,
    mode: str,
) -> None:
    """Append one JSON line per sentence parse to *output_path*."""
    if not parses:
        return
    with open(output_path, "a") as f:
        for p in parses:
            if p.edge is None:
                continue
            rec = _build_record(p, language, mode)
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# Iteration type re-export so /genparse renderers in other modules can
# accept the same type without importing trace internals.
__all__ = ["RuleIteration", "_make_genparse_command"]
