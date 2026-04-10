"""Tests for AlphaBetaParser class (using mocks to avoid loading models)."""

from unittest.mock import MagicMock, patch

import pytest
from hyperbase import hedge
from hyperbase.hyperedge import UniqueAtom

from hyperbase_parser_ab.parser import AlphaBetaParser


class TestParserInitErrors:
    def test_unsupported_language_raises(self):
        with pytest.raises(RuntimeError, match="not recognized"):
            AlphaBetaParser({"lang": "xx"})

    def test_unknown_beta_stage_raises(self):
        with (
            patch(
                "hyperbase_parser_ab.parser.SPACY_MODELS", {"en": ["en_core_web_sm"]}
            ),
            patch("spacy.util.is_package", return_value=True),
            patch("spacy.load", return_value=MagicMock()),
            patch("hyperbase_parser_ab.parser.Alpha"),
            pytest.raises(RuntimeError, match="unknown beta stage"),
        ):
            AlphaBetaParser({"lang": "en", "beta": "invalid"})

    def test_no_spacy_model_installed_raises(self):
        with (
            patch(
                "hyperbase_parser_ab.parser.SPACY_MODELS", {"en": ["en_core_web_trf"]}
            ),
            patch("spacy.util.is_package", return_value=False),
            patch("hyperbase_parser_ab.parser.Alpha"),
            pytest.raises(RuntimeError, match="requires one of the following"),
        ):
            AlphaBetaParser({"lang": "en"})


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
                "lang": "en",
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
        assert parser.normalise is True
        assert parser.post_process is True
        assert parser.debug is False
        assert parser.beta == "repair"

    def test_strict_mode(self):
        parser = _make_parser(beta="strict")
        assert parser.beta == "strict"
        # strict rules are different from repair rules
        from hyperbase_parser_ab.rules import strict_rules

        assert parser.rules is strict_rules


class TestParsernormalise:
    def test_normalise_modifier_on_predicate(self):
        """When a modifier wraps a predicate relation, move modifier to inner connector."""
        parser = _make_parser()
        # (M (P C C)) should become ((M P) C C)
        edge = hedge("(quickly/M (runs/Pd/en cat/Cc/en dog/Cc/en))")
        result = parser._normalise(edge)
        assert result
        # The modifier should be merged with the inner predicate connector
        assert result[0].not_atom  # connector should be (quickly/M runs/Pd/en)
        assert str(result[0][0]) == "quickly/M"
        assert str(result[0][1]) == "runs/Pd/en"

    def test_normalise_atom_unchanged(self):
        parser = _make_parser()
        atom = hedge("cat/Cc/en")
        assert parser._normalise(atom) == atom

    def test_normalise_non_modifier_unchanged(self):
        parser = _make_parser()
        edge = hedge("(runs/Pd/en cat/Cc/en dog/Cc/en)")
        result = parser._normalise(edge)
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
        assert parser._relation_arg_role(edge) == "r"


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
        """Equal depths → 'mm'."""
        parser = _make_parser()
        edge = hedge("(of/Br/en paris/Cp/en france/Cp/en)")
        self._setup_edge_with_depths(parser, edge, {"of": 1, "paris": 2, "france": 2})
        assert parser._builder_arg_roles(edge) == "mm"


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


class TestParserReset:
    def test_reset_clears_state(self):
        parser = _make_parser()
        parser.temp_atoms.add("dummy")
        parser.orig_atom["dummy"] = "dummy"
        parser.reset("new text")
        assert parser.cur_text == "new text"
        assert len(parser.temp_atoms) == 0
        assert len(parser.orig_atom) == 0
        assert len(parser.edge2toks) == 0
        assert len(parser.toks2edge) == 0
