"""Tests for AlphaBetaParser class (using mocks to avoid loading models)."""

import pytest
from unittest.mock import patch, MagicMock

from hyperbase.hyperedge import hedge, UniqueAtom, build_atom

from hyperbase_parser_ab.parser import AlphaBetaParser


class TestParserInitErrors:
    def test_unsupported_language_raises(self):
        with pytest.raises(RuntimeError, match="not recognized"):
            AlphaBetaParser('xx')

    def test_unknown_beta_stage_raises(self):
        with patch('hyperbase_parser_ab.parser.get_spacy_models', return_value=['en_core_web_sm']), \
             patch('spacy.util.is_package', return_value=True), \
             patch('spacy.load', return_value=MagicMock()), \
             patch('hyperbase_parser_ab.parser.Alpha'):
            with pytest.raises(RuntimeError, match='unkown beta stage'):
                AlphaBetaParser('en', beta='invalid')

    def test_no_spacy_model_installed_raises(self):
        with patch('hyperbase_parser_ab.parser.get_spacy_models', return_value=['en_core_web_trf']), \
             patch('spacy.util.is_package', return_value=False), \
             patch('hyperbase_parser_ab.parser.Alpha'):
            with pytest.raises(RuntimeError, match="requires one of the following"):
                AlphaBetaParser('en')


def _make_parser(beta='repair'):
    """Create a parser with mocked dependencies."""
    with patch('hyperbase_parser_ab.parser.get_spacy_models', return_value=['en_core_web_sm']), \
         patch('spacy.util.is_package', return_value=True), \
         patch('spacy.load', return_value=MagicMock()), \
         patch('hyperbase_parser_ab.parser.Alpha'):
        parser = AlphaBetaParser('en', beta=beta, normalize=True, post_process=True, debug=False)
    return parser


class TestParserConfig:
    def test_default_config(self):
        parser = _make_parser()
        assert parser.lang == 'en'
        assert parser.normalize is True
        assert parser.post_process is True
        assert parser.debug is False
        assert parser.beta == 'repair'

    def test_strict_mode(self):
        parser = _make_parser(beta='strict')
        assert parser.beta == 'strict'
        # strict rules are different from repair rules
        from hyperbase_parser_ab.rules import strict_rules
        assert parser.rules is strict_rules


class TestParserNormalize:
    def test_normalize_modifier_on_predicate(self):
        """When a modifier wraps a predicate relation, move modifier to inner connector."""
        parser = _make_parser()
        # (M (P C C)) should become ((M P) C C)
        edge = hedge('(quickly/M (runs/Pd/en cat/Cc/en dog/Cc/en))')
        result = parser._normalize(edge)
        assert result
        # The modifier should be merged with the inner predicate connector
        assert result[0].not_atom  # connector should be (quickly/M runs/Pd/en)
        assert str(result[0][0]) == 'quickly/M'
        assert str(result[0][1]) == 'runs/Pd/en'

    def test_normalize_atom_unchanged(self):
        parser = _make_parser()
        atom = hedge('cat/Cc/en')
        assert parser._normalize(atom) == atom

    def test_normalize_non_modifier_unchanged(self):
        parser = _make_parser()
        edge = hedge('(runs/Pd/en cat/Cc/en dog/Cc/en)')
        result = parser._normalize(edge)
        assert result == edge


class TestParserRepair:
    def test_repair_atom_unchanged(self):
        parser = _make_parser()
        atom = hedge('cat/Cc/en')
        assert parser._repair(atom) == atom

    def test_repair_normal_edge_unchanged(self):
        parser = _make_parser()
        edge = hedge('(runs/Pd/en cat/Cc/en)')
        result = parser._repair(edge)
        assert result == edge


class TestParserRelationArgRole:
    def _make_token(self, dep):
        token = MagicMock()
        token.dep_ = dep
        return token

    def test_subject(self):
        parser = _make_parser()
        edge = hedge('cat/Cc/en')
        assert edge
        uatom = UniqueAtom(edge)
        token = self._make_token('nsubj')
        parser.atom2token = {uatom: token}
        parser.orig_atom = {uatom: uatom}
        parser.depths = {uatom: 1}
        assert parser._relation_arg_role(edge) == 's'

    def test_object(self):
        parser = _make_parser()
        edge = hedge('cat/Cc/en')
        assert edge
        uatom = UniqueAtom(edge)
        token = self._make_token('dobj')
        parser.atom2token = {uatom: token}
        parser.orig_atom = {uatom: uatom}
        parser.depths = {uatom: 1}
        assert parser._relation_arg_role(edge) == 'o'

    def test_passive_subject(self):
        parser = _make_parser()
        edge = hedge('cat/Cc/en')
        assert edge
        uatom = UniqueAtom(edge)
        token = self._make_token('nsubjpass')
        parser.atom2token = {uatom: token}
        parser.orig_atom = {uatom: uatom}
        parser.depths = {uatom: 1}
        assert parser._relation_arg_role(edge) == 'p'

    def test_indirect_object(self):
        parser = _make_parser()
        edge = hedge('cat/Cc/en')
        assert edge
        uatom = UniqueAtom(edge)
        token = self._make_token('iobj')
        parser.atom2token = {uatom: token}
        parser.orig_atom = {uatom: uatom}
        parser.depths = {uatom: 1}
        assert parser._relation_arg_role(edge) == 'i'

    def test_specifier(self):
        parser = _make_parser()
        edge = hedge('cat/Cc/en')
        assert edge
        uatom = UniqueAtom(edge)
        token = self._make_token('prep')
        parser.atom2token = {uatom: token}
        parser.orig_atom = {uatom: uatom}
        parser.depths = {uatom: 1}
        assert parser._relation_arg_role(edge) == 'x'

    def test_unknown_dep(self):
        parser = _make_parser()
        edge = hedge('cat/Cc/en')
        assert edge
        uatom = UniqueAtom(edge)
        token = self._make_token('unknown_dep')
        parser.atom2token = {uatom: token}
        parser.orig_atom = {uatom: uatom}
        parser.depths = {uatom: 1}
        assert parser._relation_arg_role(edge) == '?'

    def test_clausal_complement(self):
        parser = _make_parser()
        edge = hedge('go/P/en')
        assert edge
        uatom = UniqueAtom(edge)
        token = self._make_token('xcomp')
        parser.atom2token = {uatom: token}
        parser.orig_atom = {uatom: uatom}
        parser.depths = {uatom: 1}
        assert parser._relation_arg_role(edge) == 'r'


class TestParserBuilderArgRoles:
    @staticmethod
    def _setup_edge_with_depths(parser, edge, depth_map):
        """Set up parser state using the actual atom objects from the edge."""
        from hyperbase.hyperedge import unique
        atoms = edge.all_atoms()
        for atom in atoms:
            uatom = unique(atom)
            label = str(atom).split('/')[0]
            if label in depth_map:
                parser.atom2token[uatom] = MagicMock()
                parser.orig_atom[uatom] = uatom
                parser.depths[uatom] = depth_map[label]

    def test_main_before_arg(self):
        """Lower depth first → 'ma'."""
        parser = _make_parser()
        edge = hedge('(of/Br/en paris/Cp/en france/Cp/en)')
        self._setup_edge_with_depths(parser, edge, {'of': 2, 'paris': 1, 'france': 3})
        assert parser._builder_arg_roles(edge) == 'ma'

    def test_arg_before_main(self):
        """Lower depth second → 'am'."""
        parser = _make_parser()
        edge = hedge('(of/Br/en paris/Cp/en france/Cp/en)')
        self._setup_edge_with_depths(parser, edge, {'of': 2, 'paris': 3, 'france': 1})
        assert parser._builder_arg_roles(edge) == 'am'

    def test_equal_depth(self):
        """Equal depths → 'mm'."""
        parser = _make_parser()
        edge = hedge('(of/Br/en paris/Cp/en france/Cp/en)')
        self._setup_edge_with_depths(parser, edge, {'of': 1, 'paris': 2, 'france': 2})
        assert parser._builder_arg_roles(edge) == 'mm'


class TestParserDebug:
    def test_debug_msg_when_enabled(self, capsys):
        parser = _make_parser()
        parser.debug = True
        parser.debug_msg('test message')
        assert 'test message' in capsys.readouterr().out

    def test_debug_msg_when_disabled(self, capsys):
        parser = _make_parser()
        parser.debug = False
        # Clear any output from parser construction
        capsys.readouterr()
        parser.debug_msg('test message')
        assert capsys.readouterr().out == ''


class TestParserReset:
    def test_reset_clears_state(self):
        parser = _make_parser()
        parser.temp_atoms.add('dummy')
        parser.orig_atom['dummy'] = 'dummy'
        parser.reset('new text')
        assert parser.cur_text == 'new text'
        assert len(parser.temp_atoms) == 0
        assert len(parser.orig_atom) == 0
        assert len(parser.edge2toks) == 0
        assert len(parser.toks2edge) == 0
