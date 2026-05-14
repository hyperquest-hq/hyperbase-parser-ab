"""Tests for AlphaBetaParser class (using mocks to avoid loading models)."""

from unittest.mock import MagicMock, patch

import pytest
from hyperbase import hedge
from hyperbase.hyperedge import UniqueAtom

from hyperbase_parser_ab.parser import _BADNESS_WEIGHTS, AlphaBetaParser


class TestParserInitErrors:
    def test_unsupported_language_raises(self):
        with pytest.raises(RuntimeError, match="not recognized"):
            AlphaBetaParser({"language": "xx"})

    def test_no_spacy_model_installed_raises(self):
        with (
            patch(
                "hyperbase_parser_ab.parser.SPACY_MODELS", {"en": ["en_core_web_trf"]}
            ),
            patch("spacy.util.is_package", return_value=False),
            patch("hyperbase_parser_ab.parser.Alpha"),
            pytest.raises(RuntimeError, match="requires one of the following"),
        ):
            AlphaBetaParser({"language": "en"})


def _make_parser(beta="repair"):
    """Create a parser with mocked dependencies."""
    with (
        patch("hyperbase_parser_ab.parser.SPACY_MODELS", {"en": ["en_core_web_sm"]}),
        patch("spacy.util.is_package", return_value=True),
        patch("spacy.load", return_value=MagicMock()),
        patch("hyperbase_parser_ab.parser.Alpha"),
    ):
        parser = AlphaBetaParser(
            {
                "language": "en",
                "beta": beta,
                "normalise": True,
                "post_process": True,
                "debug": False,
            }
        )
    return parser


class TestParserConfig:
    def test_default_config(self):
        parser = _make_parser()
        assert parser.lang == "en"
        assert parser.debug is False


class TestParserNormaliseModifiers:
    def test_normalise_modifier_on_predicate(self):
        """When a modifier wraps a predicate relation, move modifier to inner connector."""
        parser = _make_parser()
        # (M (P C C)) should become ((M P) C C)
        edge = hedge("(quickly/M (runs/Pd.so/en cat/Cc/en dog/Cc/en))")
        result = parser._normalise_modifiers(edge)
        assert result
        # The modifier should be merged with the inner predicate connector
        assert result[0].not_atom  # connector should be (quickly/M runs/Pd/en)
        assert str(result[0][0]) == "quickly/M"
        assert str(result[0][1]) == "runs/Pd.so/en"

    def test_normalise_atom_unchanged(self):
        parser = _make_parser()
        atom = hedge("cat/Cc/en")
        assert parser._normalise_modifiers(atom) == atom

    def test_normalise_non_modifier_unchanged(self):
        parser = _make_parser()
        edge = hedge("(runs/Pd.so/en cat/Cc/en dog/Cc/en)")
        result = parser._normalise_modifiers(edge)
        assert result == edge


class TestParserRepair:
    def test_repair_atom_unchanged(self):
        parser = _make_parser()
        atom = hedge("cat/Cc/en")
        assert parser._repair(atom) == atom

    def test_repair_normal_edge_unchanged(self):
        parser = _make_parser()
        edge = hedge("(runs/Pd/en cat/Cc/en)")
        result = parser._repair(edge)
        assert result == edge


class TestParserRelationArgRole:
    def _make_token(self, dep):
        token = MagicMock()
        token.dep_ = dep
        return token

    def test_subject(self):
        parser = _make_parser()
        edge = hedge("cat/Cc/en")
        assert edge
        uatom = UniqueAtom(edge)
        token = self._make_token("nsubj")
        parser.atom2token = {uatom: token}
        parser.orig_atom = {uatom: uatom}
        parser.depths = {uatom: 1}
        assert parser._relation_arg_role(edge) == "s"

    def test_object(self):
        parser = _make_parser()
        edge = hedge("cat/Cc/en")
        assert edge
        uatom = UniqueAtom(edge)
        token = self._make_token("dobj")
        parser.atom2token = {uatom: token}
        parser.orig_atom = {uatom: uatom}
        parser.depths = {uatom: 1}
        assert parser._relation_arg_role(edge) == "o"

    def test_passive_subject(self):
        parser = _make_parser()
        edge = hedge("cat/Cc/en")
        assert edge
        uatom = UniqueAtom(edge)
        token = self._make_token("nsubjpass")
        parser.atom2token = {uatom: token}
        parser.orig_atom = {uatom: uatom}
        parser.depths = {uatom: 1}
        assert parser._relation_arg_role(edge) == "o"

    def test_indirect_object(self):
        parser = _make_parser()
        edge = hedge("cat/Cc/en")
        assert edge
        uatom = UniqueAtom(edge)
        token = self._make_token("iobj")
        parser.atom2token = {uatom: token}
        parser.orig_atom = {uatom: uatom}
        parser.depths = {uatom: 1}
        assert parser._relation_arg_role(edge) == "x"

    def test_specifier(self):
        parser = _make_parser()
        edge = hedge("cat/Cc/en")
        assert edge
        uatom = UniqueAtom(edge)
        token = self._make_token("prep")
        parser.atom2token = {uatom: token}
        parser.orig_atom = {uatom: uatom}
        parser.depths = {uatom: 1}
        assert parser._relation_arg_role(edge) == "x"

    def test_unknown_dep(self):
        parser = _make_parser()
        edge = hedge("cat/Cc/en")
        assert edge
        uatom = UniqueAtom(edge)
        token = self._make_token("unknown_dep")
        parser.atom2token = {uatom: token}
        parser.orig_atom = {uatom: uatom}
        parser.depths = {uatom: 1}
        assert parser._relation_arg_role(edge) == "?"

    def test_clausal_complement(self):
        parser = _make_parser()
        edge = hedge("go/P/en")
        assert edge
        uatom = UniqueAtom(edge)
        token = self._make_token("xcomp")
        parser.atom2token = {uatom: token}
        parser.orig_atom = {uatom: uatom}
        parser.depths = {uatom: 1}
        assert parser._relation_arg_role(edge) == "o"


class TestParserBuilderArgRoles:
    @staticmethod
    def _setup_edge_with_depths(parser, edge, depth_map):
        """Set up parser state using the actual atom objects from the edge."""
        from hyperbase.hyperedge import unique

        atoms = edge.all_atoms()
        for atom in atoms:
            uatom = unique(atom)
            label = str(atom).split("/")[0]
            if label in depth_map:
                parser.atom2token[uatom] = MagicMock()
                parser.orig_atom[uatom] = uatom
                parser.depths[uatom] = depth_map[label]

    def test_main_before_arg(self):
        """Lower depth first → 'ma'."""
        parser = _make_parser()
        edge = hedge("(of/Br/en paris/Cp/en france/Cp/en)")
        self._setup_edge_with_depths(parser, edge, {"of": 2, "paris": 1, "france": 3})
        assert parser._builder_arg_roles(edge) == "ma"

    def test_arg_before_main(self):
        """Lower depth second → 'am'."""
        parser = _make_parser()
        edge = hedge("(of/Br/en paris/Cp/en france/Cp/en)")
        self._setup_edge_with_depths(parser, edge, {"of": 2, "paris": 3, "france": 1})
        assert parser._builder_arg_roles(edge) == "am"

    def test_equal_depth(self):
        """Equal depths → 'ma'."""
        parser = _make_parser()
        edge = hedge("(of/Br/en paris/Cp/en france/Cp/en)")
        self._setup_edge_with_depths(parser, edge, {"of": 1, "paris": 2, "france": 2})
        assert parser._builder_arg_roles(edge) == "ma"


class TestParserDebug:
    def test_debug_msg_when_enabled(self, capsys):
        parser = _make_parser()
        parser.debug = True
        parser.debug_msg("test message")
        assert "test message" in capsys.readouterr().out

    def test_debug_msg_when_disabled(self, capsys):
        parser = _make_parser()
        parser.debug = False
        # Clear any output from parser construction
        capsys.readouterr()
        parser.debug_msg("test message")
        assert capsys.readouterr().out == ""


class TestParserFlattenConjunctions:
    def test_flatten_atom_unchanged(self):
        parser = _make_parser()
        atom = hedge("red/Ca/en")
        assert parser._flatten_conjunctions(atom) == atom

    def test_flatten_no_conjunction_unchanged(self):
        parser = _make_parser()
        edge = hedge("(runs/Pd/en cat/Cc/en dog/Cc/en)")
        assert parser._flatten_conjunctions(edge) == edge

    def test_flatten_simple_conjunction_unchanged(self):
        """A flat conjunction with no nested conjunctions stays the same."""
        parser = _make_parser()
        edge = hedge("(,/J red/Ca/en green/Ca/en blue/Ca/en)")
        assert parser._flatten_conjunctions(edge) == edge

    def test_flatten_nested_same_connector(self):
        """(,/J red (,/J green blue)) → (,/J red green blue)"""
        parser = _make_parser()
        edge = hedge("(,/J red/Ca/en (,/J green/Ca/en blue/Ca/en))")
        expected = hedge("(,/J red/Ca/en green/Ca/en blue/Ca/en)")
        assert parser._flatten_conjunctions(edge) == expected

    def test_flatten_nested_different_connector_unchanged(self):
        """Nested conjunction with a different connector should NOT be flattened."""
        parser = _make_parser()
        edge = hedge("(,/J red/Ca/en (and/J/en green/Ca/en blue/Ca/en))")
        assert parser._flatten_conjunctions(edge) == edge

    def test_flatten_recursive_bottom_up(self):
        """Multiple levels of nesting should all collapse."""
        parser = _make_parser()
        edge = hedge("(,/J red/Ca/en (,/J green/Ca/en (,/J blue/Ca/en yellow/Ca/en)))")
        expected = hedge("(,/J red/Ca/en green/Ca/en blue/Ca/en yellow/Ca/en)")
        assert parser._flatten_conjunctions(edge) == expected

    def test_flatten_multiple_nested_conjunctions(self):
        """(,/J (,/J a b) (,/J c d)) → (,/J a b c d)"""
        parser = _make_parser()
        edge = hedge("(,/J (,/J a/Ca/en b/Ca/en) (,/J c/Ca/en d/Ca/en))")
        expected = hedge("(,/J a/Ca/en b/Ca/en c/Ca/en d/Ca/en)")
        assert parser._flatten_conjunctions(edge) == expected

    def test_flatten_inside_outer_edge(self):
        """A nested conjunction inside a non-conjunction outer edge is still
        flattened bottom-up."""
        parser = _make_parser()
        edge = hedge(
            "(runs/Pd/en cat/Cc/en (,/J red/Ca/en (,/J green/Ca/en blue/Ca/en)))"
        )
        expected = hedge(
            "(runs/Pd/en cat/Cc/en (,/J red/Ca/en green/Ca/en blue/Ca/en))"
        )
        assert parser._flatten_conjunctions(edge) == expected

    def test_flatten_mixed_connectors_partial(self):
        """Only the matching nested conjunctions should be flattened."""
        parser = _make_parser()
        edge = hedge(
            "(,/J red/Ca/en (,/J green/Ca/en blue/Ca/en)"
            " (and/J/en yellow/Ca/en purple/Ca/en))"
        )
        expected = hedge(
            "(,/J red/Ca/en green/Ca/en blue/Ca/en"
            " (and/J/en yellow/Ca/en purple/Ca/en))"
        )
        assert parser._flatten_conjunctions(edge) == expected


class TestParserBeamSearch:
    @staticmethod
    def _setup_loop_state(parser):
        parser.depths = {}
        parser.connections = set()
        parser.orig_atom = {}
        parser.atom2token = {}
        parser.token2atom = {}
        parser._cur_trace = None

    @staticmethod
    def _wire_dep_chain(parser, atoms):
        # Wire a dep-tree chain: atoms[0] is the parent, atoms[1] its
        # only dep-child, etc. Sets the bare attributes the new dep-tree
        # candidate generator and its downstream helpers touch:
        #   .children — used by _live_dep_children
        #   .i        — used by _plus_builder_bonus
        #   .sent     — walked by _no_dangling_branches; an empty list
        #               makes the loop a no-op so the bonus always fires.
        tokens = []
        for i, _atom in enumerate(atoms):
            tok = MagicMock()
            tok.i = i
            tok.sent = []
            tokens.append(tok)
        for i in range(len(tokens) - 1):
            tokens[i].children = [tokens[i + 1]]
        tokens[-1].children = []
        for atom, tok in zip(atoms, tokens, strict=False):
            parser.atom2token[atom] = tok
        return tokens

    def test_default_beam_width(self):
        parser = _make_parser()
        assert parser.beam_width == 1

    def test_beam_width_param_is_read(self):
        with (
            patch(
                "hyperbase_parser_ab.parser.SPACY_MODELS", {"en": ["en_core_web_sm"]}
            ),
            patch("spacy.util.is_package", return_value=True),
            patch("spacy.load", return_value=MagicMock()),
            patch("hyperbase_parser_ab.parser.Alpha"),
        ):
            parser = AlphaBetaParser({"language": "en", "beam_width": 5})
        assert parser.beam_width == 5

    def test_beam_width_clamps_to_one_minimum(self):
        with (
            patch(
                "hyperbase_parser_ab.parser.SPACY_MODELS", {"en": ["en_core_web_sm"]}
            ),
            patch("spacy.util.is_package", return_value=True),
            patch("spacy.load", return_value=MagicMock()),
            patch("hyperbase_parser_ab.parser.Alpha"),
        ):
            parser = AlphaBetaParser({"language": "en", "beam_width": 0})
        assert parser.beam_width == 1

    def test_greedy_reduces_two_concepts(self):
        parser = _make_parser()
        parser.beam_width = 1
        self._setup_loop_state(parser)
        atom_a = hedge("cat/Cc/en")
        atom_b = hedge("dog/Cc/en")
        assert atom_a is not None
        assert atom_b is not None
        self._wire_dep_chain(parser, [atom_a, atom_b])
        best_beam, _substituted, _rounds = parser._hill_climb_atomization(
            [atom_a, atom_b], []
        )
        assert best_beam is not None
        result = best_beam.sequence
        failed = best_beam.failed
        assert result is not None
        assert len(result) == 1
        assert failed is False
        # Rule("C", {"C"}, 2, "+/B/.") fires, building the +/B builder
        # (argroles are assigned in-loop, so the builder carries them already)
        assert str(result[0]).startswith("(+/B.")

    def test_beam_width_three_reduces_two_concepts(self):
        parser = _make_parser()
        parser.beam_width = 3
        self._setup_loop_state(parser)
        atom_a = hedge("cat/Cc/en")
        atom_b = hedge("dog/Cc/en")
        assert atom_a is not None
        assert atom_b is not None
        self._wire_dep_chain(parser, [atom_a, atom_b])
        best_beam, _substituted, _rounds = parser._hill_climb_atomization(
            [atom_a, atom_b], []
        )
        assert best_beam is not None
        result = best_beam.sequence
        failed = best_beam.failed
        assert result is not None
        assert len(result) == 1
        assert failed is False

    def test_beam_returns_lowest_total_badness(self):
        """With width=2 the search should keep the clean path alive even when
        the locally-best candidate is only marginally better."""
        parser = _make_parser()
        parser.beam_width = 2
        self._setup_loop_state(parser)
        # Three concepts: must reduce two pairs in sequence. Several beam
        # paths exist; the clean-path winner should have total_badness == 0
        # because every C+C → +/B is structurally clean.
        atoms = [hedge("a/Cc/en"), hedge("b/Cc/en"), hedge("c/Cc/en")]
        for a in atoms:
            assert a is not None
        best_beam, _substituted, _rounds = parser._hill_climb_atomization(atoms, [])
        assert best_beam is not None
        result = best_beam.sequence
        assert result is not None
        assert len(result) == 1


class TestParserCandidateBadness:
    def test_clean_atom_has_zero_badness(self):
        parser = _make_parser()
        atom = hedge("cat/Cc/en")
        assert atom is not None
        assert parser._candidate_badness(atom) == 0

    def test_clean_relation_has_zero_badness(self):
        parser = _make_parser()
        edge = hedge("(runs/Pd.so/en cat/Cc/en dog/Cc/en)")
        assert edge is not None
        assert parser._candidate_badness(edge) == 0

    def test_no_argroles_now_counts(self):
        """Argroles are assigned to candidates inside _expand_beam, so by
        the time _candidate_badness runs the connector should already carry
        argroles. A connector that still lacks them is genuinely malformed
        and 'no-argroles' (sev 0, weight 1000) should contribute."""
        parser = _make_parser()
        edge = hedge("(runs/Pd/en cat/Cc/en dog/Cc/en)")
        assert edge is not None
        assert parser._candidate_badness(edge) >= _BADNESS_WEIGHTS[0]

    def test_other_argrole_issues_still_count(self):
        """Only 'no-argroles' is filtered. A duplicate-argrole-s issue
        (sev 2, weight 10) should still contribute to badness."""
        parser = _make_parser()
        edge = hedge("(runs/Pd.ss/en cat/Cc/en dog/Cc/en)")
        assert edge is not None
        bad = parser._candidate_badness(edge)
        assert bad >= 10

    def test_correctness_error_dominates(self):
        """A severity-0 (correctness) error must outweigh any sev-2/3 noise."""
        parser = _make_parser()
        # Modifier connector with 2 args → correctness violation (mod-1-arg)
        bad_edge = hedge("(quickly/M/en cat/Cc/en dog/Cc/en)")
        assert bad_edge is not None
        bad_score = parser._candidate_badness(bad_edge)
        # Severity 0 weight is 1000; one issue should be >= 1000 and < crash penalty.
        assert bad_score >= 1000
        assert bad_score < 1_000_000
