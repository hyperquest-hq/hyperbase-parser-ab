"""Structured per-parse trace consumed by the REPL ``report`` toggle.

The parser populates a :class:`ParseTrace` only when invoked through the
REPL with ``report`` enabled; library and CLI callers never see these
dataclasses on ``ParseResult.extra``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hyperbase_parser_ab.rules import Rule


@dataclass
class AtomTrace:
    token_text: str
    token_idx: int
    predicted_type: str
    refined_type: str
    final_atom: str
    dropped: bool


@dataclass
class RuleCandidate:
    rule_index: int
    rule_repr: str
    pos: int
    score: int
    new_edge_repr: str
    is_winner: bool = False


@dataclass
class RuleIteration:
    iteration: int
    sequence_repr: list[str]
    candidates: list[RuleCandidate] = field(default_factory=list)
    rejections: list[tuple[int, int]] = field(default_factory=list)
    fallback_used: bool = False


@dataclass
class ParseTrace:
    atoms: list[AtomTrace] = field(default_factory=list)
    iterations: list[RuleIteration] = field(default_factory=list)
    post_processing: list[tuple[str, str]] = field(default_factory=list)


def rule_repr(rule: Rule, index: int) -> str:
    args: str = "{" + ",".join(sorted(rule.arg_types)) + "}"
    if rule.connector:
        return f"#{index} {rule.first_type}+{args}@{rule.size}→{rule.connector}"
    return f"#{index} {rule.first_type}+{args}@{rule.size}"
