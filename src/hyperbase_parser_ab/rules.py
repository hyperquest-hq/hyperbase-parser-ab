from collections.abc import Mapping

from hyperbase import hedge
from hyperbase.hyperedge import Atom, Hyperedge, unique
from spacy.tokens import Token


class Rule:
    def __init__(
        self,
        first_type: str,
        arg_types: set[str],
        size: int,
        connector: str | None = None,
        can_dominate: bool = True,
        mandatory: bool = False,
        dep_blockers: set[str] | None = None,
        consecutive_siblings_ok: bool = False,
        relaxed_head_satisfaction: bool = True,
    ) -> None:
        self.first_type: str = first_type
        self.arg_types: set[str] = arg_types
        self.size: int = size
        self.connector: str | None = connector
        self.can_dominate: bool = can_dominate
        self.mandatory: bool = mandatory
        self.dep_blockers: set[str] | None = dep_blockers
        self.consecutive_siblings_ok: bool = consecutive_siblings_ok
        self.relaxed_head_satisfaction: bool = relaxed_head_satisfaction
        self._branches: int = 0


RULES: list[Rule] = [
    Rule("C", {"C"}, 2, "+/B/.", mandatory=True, dep_blockers={"conj"}),
    Rule(
        "C",
        {"C", "R"},
        2,
        ":/J/.",
        consecutive_siblings_ok=True,
        relaxed_head_satisfaction=True,
    ),
    # Rule("M", {"C", "R", "M", "S", "T", "P", "B", "J"}, 2),
    Rule("M", {"C", "M", "T", "P"}, 2),
    Rule("B", {"C"}, 3),
    Rule("T", {"C", "R"}, 2),
    Rule("P", {"C", "R", "S"}, 6),
    Rule("P", {"C", "R", "S"}, 5),
    Rule("P", {"C", "R", "S"}, 4),
    Rule("P", {"C", "R", "S"}, 3),
    Rule("P", {"C", "R", "S"}, 2),
    # Rule("J", {"C", "R", "M", "S", "T", "P", "B", "J"}, 3, can_dominate=False),
    # Rule("J", {"C", "R", "M", "S", "T", "P", "B", "J"}, 2, can_dominate=False),
    Rule("J", {"R"}, 3, can_dominate=False),
    Rule("J", {"C"}, 3, can_dominate=False),
    # Rule("J", {"C", "R"}, 2, can_dominate=False),
]


def _window_has_blocked_dep_indices(
    sentence: list[Hyperedge],
    indices: list[int],
    dep_blockers: set[str],
    atom2token: Mapping[Atom, Token],
) -> bool:
    for seq_idx in indices:
        edge: Hyperedge = sentence[seq_idx]
        for atom in edge.all_atoms():
            uatom = unique(atom)
            if not isinstance(uatom, Atom):
                continue
            token = atom2token.get(uatom)
            if token is not None and token.dep_ in dep_blockers:
                return True
    return False


def _window_has_blocked_dep(
    sentence: list[Hyperedge],
    pos: int,
    size: int,
    dep_blockers: set[str],
    atom2token: Mapping[Atom, Token],
) -> bool:
    return _window_has_blocked_dep_indices(
        sentence,
        list(range(pos - size + 1, pos + 1)),
        dep_blockers,
        atom2token,
    )


def apply_rule_indices(
    rule: Rule,
    sentence: list[Hyperedge],
    indices: list[int],
    atom2token: Mapping[Atom, Token] | None = None,
) -> Hyperedge | None:
    if len(indices) != rule.size:
        return None
    if (
        rule.dep_blockers
        and atom2token is not None
        and _window_has_blocked_dep_indices(
            sentence, indices, rule.dep_blockers, atom2token
        )
    ):
        return None
    for pivot_pos in range(rule.size):
        args: list[Hyperedge] = []
        pivot: Hyperedge | None = None
        valid: bool = True
        for i, seq_idx in enumerate(indices):
            edge: Hyperedge = sentence[seq_idx]
            if i == pivot_pos:
                if edge.mtype() == rule.first_type:
                    if rule.connector:
                        args.append(edge)
                    else:
                        pivot = edge
                else:
                    valid = False
                    break
            else:
                if edge.mtype() in rule.arg_types:
                    args.append(edge)
                else:
                    valid = False
                    break
        if valid:
            if rule.connector:
                return hedge([rule.connector, *args])
            else:
                return hedge([pivot, *args])
    return None


def apply_rule(
    rule: Rule,
    sentence: list[Hyperedge],
    pos: int,
    atom2token: Mapping[Atom, Token] | None = None,
) -> Hyperedge | None:
    return apply_rule_indices(
        rule,
        sentence,
        list(range(pos - rule.size + 1, pos + 1)),
        atom2token,
    )
