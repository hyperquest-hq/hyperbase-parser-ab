import pytest

from hyperbase.hyperedge import hedge
from hyperbase_parser_ab.rules import Rule, apply_rule, strict_rules, repair_rules


class TestRule:
    def test_rule_attributes(self):
        rule = Rule('P', {'C', 'R'}, 3)
        assert rule.first_type == 'P'
        assert rule.arg_types == {'C', 'R'}
        assert rule.size == 3
        assert rule.connector is None

    def test_rule_with_connector(self):
        rule = Rule('C', {'C'}, 2, '+/B/.')
        assert rule.connector == '+/B/.'

    def test_strict_rules_count(self):
        assert len(strict_rules) == 10

    def test_repair_rules_count(self):
        assert len(repair_rules) == 11

    def test_repair_has_extra_conjunction_rule(self):
        """Repair rules should have a 2-arg conjunction rule that strict doesn't."""
        repair_j_rules = [r for r in repair_rules if r.first_type == 'J']
        strict_j_rules = [r for r in strict_rules if r.first_type == 'J']
        assert len(repair_j_rules) == 2
        assert len(strict_j_rules) == 1

    def test_repair_builder_accepts_relations(self):
        """Repair builder rule should accept R in addition to C."""
        repair_b_rules = [r for r in repair_rules if r.first_type == 'B']
        assert 'R' in repair_b_rules[0].arg_types

    def test_strict_builder_only_concepts(self):
        """Strict builder rule should only accept C."""
        strict_b_rules = [r for r in strict_rules if r.first_type == 'B']
        assert strict_b_rules[0].arg_types == {'C'}


class TestApplyRule:
    def test_concept_concept_builder(self):
        """Two concepts should combine with a builder connector."""
        rule = Rule('C', {'C'}, 2, '+/B/.')
        c1 = hedge('cat/Cc/en')
        c2 = hedge('dog/Cc/en')
        result = apply_rule(rule, [c1, c2], 1)
        assert result is not None
        assert str(result[0]) == '+/B/.'

    def test_modifier_applied_to_concept(self):
        """Modifier + concept should compose."""
        rule = Rule('M', {'C', 'R', 'M', 'S', 'T', 'P', 'B', 'J'}, 2)
        mod = hedge('red/Ma/en')
        concept = hedge('car/Cc/en')
        result = apply_rule(rule, [mod, concept], 1)
        assert result is not None
        # modifier becomes the connector
        assert str(result[0]) == 'red/Ma/en'
        assert str(result[1]) == 'car/Cc/en'

    def test_concept_modifier_order(self):
        """Concept + modifier should also compose (modifier as pivot)."""
        rule = Rule('M', {'C', 'R', 'M', 'S', 'T', 'P', 'B', 'J'}, 2)
        concept = hedge('car/Cc/en')
        mod = hedge('red/Ma/en')
        result = apply_rule(rule, [concept, mod], 1)
        assert result is not None

    def test_predicate_with_two_args(self):
        rule = Rule('P', {'C', 'R', 'S'}, 3)
        pred = hedge('likes/Pd/en')
        subj = hedge('mary/Cp/en')
        obj = hedge('cake/Cc/en')
        result = apply_rule(rule, [subj, pred, obj], 2)
        assert result is not None
        assert str(result[0]) == 'likes/Pd/en'

    def test_no_match_wrong_type(self):
        """Rule should not apply when types don't match."""
        rule = Rule('P', {'C'}, 2)
        c1 = hedge('cat/Cc/en')
        c2 = hedge('dog/Cc/en')
        # Neither is a predicate, so rule shouldn't match
        result = apply_rule(rule, [c1, c2], 1)
        assert result is None

    def test_no_match_wrong_arg_type(self):
        """Rule should not apply when argument types don't match."""
        rule = Rule('B', {'C'}, 3)  # strict: only concepts
        builder = hedge('of/Br/en')
        concept = hedge('cat/Cc/en')
        pred = hedge('runs/Pd/en')  # P not in {C}
        result = apply_rule(rule, [concept, builder, pred], 2)
        assert result is None

    def test_conjunction_three_args(self):
        rule = Rule('J', {'C', 'R', 'M', 'S', 'T', 'P', 'B', 'J'}, 3)
        conj = hedge('and/J/en')
        c1 = hedge('cat/Cc/en')
        c2 = hedge('dog/Cc/en')
        result = apply_rule(rule, [c1, conj, c2], 2)
        assert result is not None
        assert str(result[0]) == 'and/J/en'

    def test_trigger_with_concept(self):
        rule = Rule('T', {'C', 'R'}, 2)
        trigger = hedge('the/Td/en')
        concept = hedge('cat/Cc/en')
        result = apply_rule(rule, [trigger, concept], 1)
        assert result is not None
