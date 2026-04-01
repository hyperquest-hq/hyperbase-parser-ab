from hyperbase.hyperedge import Hyperedge, hedge


class Rule:
    def __init__(self, first_type: str, arg_types: set[str], size: int, connector: str | None = None) -> None:
        self.first_type: str = first_type
        self.arg_types: set[str] = arg_types
        self.size: int = size
        self.connector: str | None = connector
        self._branches: int = 0


strict_rules: list[Rule] = [
    Rule('C', {'C'}, 2, '+/B/.'),
    Rule('M', {'C', 'R', 'M', 'S', 'T', 'P', 'B', 'J'}, 2),
    Rule('B', {'C'}, 3),
    Rule('T', {'C', 'R'}, 2),
    Rule('P', {'C', 'R', 'S'}, 6),
    Rule('P', {'C', 'R', 'S'}, 5),
    Rule('P', {'C', 'R', 'S'}, 4),
    Rule('P', {'C', 'R', 'S'}, 3),
    Rule('P', {'C', 'R', 'S'}, 2),
    Rule('J', {'C', 'R', 'M', 'S', 'T', 'P', 'B', 'J'}, 3)]


repair_rules: list[Rule] = [
    Rule('C', {'C'}, 2, '+/B/.'),
    Rule('M', {'C', 'R', 'M', 'S', 'T', 'P', 'B', 'J'}, 2),
    Rule('B', {'C', 'R'}, 3),
    Rule('T', {'C', 'R'}, 2),
    Rule('P', {'C', 'R', 'S'}, 6),
    Rule('P', {'C', 'R', 'S'}, 5),
    Rule('P', {'C', 'R', 'S'}, 4),
    Rule('P', {'C', 'R', 'S'}, 3),
    Rule('P', {'C', 'R', 'S'}, 2),
    Rule('J', {'C', 'R', 'M', 'S', 'T', 'P', 'B', 'J'}, 3),
    Rule('J', {'C', 'R', 'M', 'S', 'T', 'P', 'B', 'J'}, 2)]


def apply_rule(rule: Rule, sentence: list[Hyperedge], pos: int) -> Hyperedge | None:
    for pivot_pos in range(rule.size):
        args: list[Hyperedge] = []
        pivot: Hyperedge | None = None
        valid: bool = True
        for i in range(rule.size):
            edge: Hyperedge = sentence[pos - rule.size + i + 1]
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
                return hedge([rule.connector] + args)
            else:
                return hedge([pivot] + args)
    return None
