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
    top_candidates: list[tuple[str, float]] = field(default_factory=list)
    chosen_label_rank: int = 0
    is_uncertain: bool = False


@dataclass
class RuleCandidate:
    rule_index: int
    rule_repr: str
    pos: int
    score: int
    new_edge_repr: str
    badness: int = 0
    distortion: int = 0
    is_winner: bool = False
    indices: list[int] | None = None


@dataclass
class RuleIteration:
    iteration: int
    sequence_repr: list[str]
    candidates: list[RuleCandidate] = field(default_factory=list)
    dominated: list[RuleCandidate] = field(default_factory=list)
    fallback_used: bool = False


@dataclass
class SubstitutionTrial:
    tok_idx: int
    token_text: str
    label_from: str
    label_to: str
    badness: int
    distortion: int
    score: int
    edge_repr: str
    is_winner: bool = False
    number: int = 0
    substitutions: dict[int, str] = field(default_factory=dict)


@dataclass
class SubstitutionRound:
    round_idx: int
    seed_badness: int
    seed_distortion: int
    seed_score: int
    trials: list[SubstitutionTrial] = field(default_factory=list)
    improved: bool = False


@dataclass
class ParseTrace:
    atoms: list[AtomTrace] = field(default_factory=list)
    iterations: list[RuleIteration] = field(default_factory=list)
    post_processing: list[tuple[str, str]] = field(default_factory=list)
    final_badness: dict[str, list[tuple[str, str, int]]] = field(default_factory=dict)
    total_badness: int = 0
    total_distortion: int = 0
    substitution_rounds: list[SubstitutionRound] = field(default_factory=list)
    # Stranded-group set detected after each pass of the orchestration
    # loop in parse_spacy_sentence. Each group is rendered as the "+"
    # join of its sorted leaf-atom strings (a singleton string for an
    # atom strand, "a+b+c" for a non-atom hyperedge strand). One entry
    # per pass actually performed, in order; len(passes) == number of
    # passes.
    passes: list[list[str]] = field(default_factory=list)


def rule_repr(rule: Rule, index: int) -> str:
    args: str = "{" + ",".join(sorted(rule.arg_types)) + "}"
    if rule.connector:
        return f"#{index} {rule.first_type}+{args}@{rule.size}→{rule.connector}"
    return f"#{index} {rule.first_type}+{args}@{rule.size}"
