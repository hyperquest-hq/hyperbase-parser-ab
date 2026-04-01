"""Tests for parser module helper functions that classify tokens by type/subtype."""

from unittest.mock import MagicMock

from hyperbase_parser_ab.parser import (
    _concept_type_and_subtype,
    _modifier_type_and_subtype,
    _builder_type_and_subtype,
    _predicate_type_and_subtype,
    _is_verb,
    _generate_tok_pos,
)
from hyperbase.hyperedge import hedge, UniqueAtom


def _mock_token(pos='NOUN', dep='nsubj', tag='NN', head_dep=None, head_pos=None):
    token = MagicMock()
    token.pos_ = pos
    token.dep_ = dep
    token.tag_ = tag
    if head_dep or head_pos:
        head = MagicMock()
        head.dep_ = head_dep or ''
        head.pos_ = head_pos or ''
        token.head = head
    return token


# ========================
# _concept_type_and_subtype
# ========================

class TestConceptTypeAndSubtype:
    def test_nmod_dependency(self):
        token = _mock_token(pos='NOUN', dep='nmod')
        assert _concept_type_and_subtype(token) == 'Cm'

    def test_adjective(self):
        token = _mock_token(pos='ADJ', dep='amod')
        assert _concept_type_and_subtype(token) == 'Ca'

    def test_noun(self):
        token = _mock_token(pos='NOUN', dep='nsubj')
        assert _concept_type_and_subtype(token) == 'Cc'

    def test_proper_noun(self):
        token = _mock_token(pos='PROPN', dep='nsubj')
        assert _concept_type_and_subtype(token) == 'Cp'

    def test_number(self):
        token = _mock_token(pos='NUM', dep='nummod')
        assert _concept_type_and_subtype(token) == 'C#'

    def test_determiner(self):
        token = _mock_token(pos='DET', dep='det')
        assert _concept_type_and_subtype(token) == 'Cd'

    def test_pronoun(self):
        token = _mock_token(pos='PRON', dep='nsubj')
        assert _concept_type_and_subtype(token) == 'Ci'

    def test_fallback(self):
        token = _mock_token(pos='X', dep='dep')
        assert _concept_type_and_subtype(token) == 'C'

    def test_nmod_takes_priority_over_pos(self):
        """nmod dependency should override any POS tag."""
        token = _mock_token(pos='ADJ', dep='nmod')
        assert _concept_type_and_subtype(token) == 'Cm'


# ==========================
# _modifier_type_and_subtype
# ==========================

class TestModifierTypeAndSubtype:
    def test_negation(self):
        token = _mock_token(pos='ADV', dep='neg')
        assert _modifier_type_and_subtype(token) == 'Mn'

    def test_nk_negation(self):
        token = _mock_token(pos='ADV', dep='nk')
        assert _modifier_type_and_subtype(token) == 'Mn'

    def test_possessive(self):
        token = _mock_token(pos='DET', dep='poss')
        assert _modifier_type_and_subtype(token) == 'Mp'

    def test_pg_possessive(self):
        token = _mock_token(pos='DET', dep='pg')
        assert _modifier_type_and_subtype(token) == 'Mp'

    def test_preposition(self):
        token = _mock_token(pos='ADP', dep='prep')
        assert _modifier_type_and_subtype(token) == 'Mt'

    def test_conjunctional(self):
        token = _mock_token(pos='CCONJ', dep='preconj')
        assert _modifier_type_and_subtype(token) == 'Mj'

    def test_adjective(self):
        token = _mock_token(pos='ADJ', dep='amod')
        assert _modifier_type_and_subtype(token) == 'Ma'

    def test_determiner(self):
        token = _mock_token(pos='DET', dep='det')
        assert _modifier_type_and_subtype(token) == 'Md'

    def test_number(self):
        token = _mock_token(pos='NUM', dep='nummod')
        assert _modifier_type_and_subtype(token) == 'M#'

    def test_auxiliary_modal(self):
        token = _mock_token(pos='AUX', dep='aux')
        assert _modifier_type_and_subtype(token) == 'Mm'

    def test_particle(self):
        token = _mock_token(pos='VERB', dep='prt')
        assert _modifier_type_and_subtype(token) == 'Ml'

    def test_infinitive_part(self):
        token = _mock_token(pos='PART', dep='mark')
        assert _modifier_type_and_subtype(token) == 'Mi'

    def test_adverb(self):
        token = _mock_token(pos='ADV', dep='advmod')
        assert _modifier_type_and_subtype(token) == 'M'

    def test_fallback(self):
        token = _mock_token(pos='X', dep='dep')
        assert _modifier_type_and_subtype(token) == 'M'


# =========================
# _builder_type_and_subtype
# =========================

class TestBuilderTypeAndSubtype:
    def test_case_dependency(self):
        token = _mock_token(pos='ADP', dep='case')
        assert _builder_type_and_subtype(token) == 'Bp'

    def test_pg_dependency(self):
        token = _mock_token(pos='ADP', dep='pg')
        assert _builder_type_and_subtype(token) == 'Bp'

    def test_ag_dependency(self):
        token = _mock_token(pos='ADP', dep='ag')
        assert _builder_type_and_subtype(token) == 'Bp'

    def test_adposition(self):
        token = _mock_token(pos='ADP', dep='prep')
        assert _builder_type_and_subtype(token) == 'Br'

    def test_determiner(self):
        token = _mock_token(pos='DET', dep='det')
        assert _builder_type_and_subtype(token) == 'Bd'

    def test_fallback(self):
        token = _mock_token(pos='CCONJ', dep='cc')
        assert _builder_type_and_subtype(token) == 'B'


# ============================
# _predicate_type_and_subtype
# ============================

class TestPredicateTypeAndSubtype:
    def test_advcl(self):
        token = _mock_token(pos='VERB', dep='advcl')
        assert _predicate_type_and_subtype(token) == 'Pd'

    def test_csubj(self):
        token = _mock_token(pos='VERB', dep='csubj')
        assert _predicate_type_and_subtype(token) == 'Pd'

    def test_parataxis(self):
        token = _mock_token(pos='VERB', dep='parataxis')
        assert _predicate_type_and_subtype(token) == 'Pd'

    def test_relcl(self):
        token = _mock_token(pos='VERB', dep='relcl')
        assert _predicate_type_and_subtype(token) == 'P'

    def test_ccomp(self):
        token = _mock_token(pos='VERB', dep='ccomp')
        assert _predicate_type_and_subtype(token) == 'P'

    def test_xcomp(self):
        token = _mock_token(pos='VERB', dep='xcomp')
        assert _predicate_type_and_subtype(token) == 'P'

    def test_verb_with_other_dep(self):
        """A VERB with an unclassified dep should be Pd."""
        token = _mock_token(pos='VERB', dep='ROOT')
        assert _predicate_type_and_subtype(token) == 'Pd'

    def test_non_verb(self):
        """Non-verb tokens default to P."""
        token = _mock_token(pos='NOUN', dep='ROOT')
        assert _predicate_type_and_subtype(token) == 'P'


# ========
# _is_verb
# ========

class TestIsVerb:
    def test_verb(self):
        token = _mock_token(pos='VERB')
        assert _is_verb(token) is True

    def test_not_verb(self):
        token = _mock_token(pos='NOUN')
        assert _is_verb(token) is False

    def test_aux_is_not_verb(self):
        token = _mock_token(pos='AUX')
        assert _is_verb(token) is False


# ================
# _generate_tok_pos
# ================

class TestGenerateTokPos:
    def test_atom_with_mapping(self):
        atom = hedge('cat/Cc/en')
        assert atom
        uatom = UniqueAtom(atom)
        atom2word = {uatom: ('cat', 0)}
        result = _generate_tok_pos(atom2word, atom)
        assert result == '0'

    def test_atom_without_mapping(self):
        atom = hedge('cat/Cc/en')
        result = _generate_tok_pos({}, atom)
        assert result == '-1'

    def test_composite_edge(self):
        edge = hedge('(red/Ma/en cat/Cc/en)')
        assert edge
        atoms = edge.all_atoms()
        atom2word = {}
        for i, atom in enumerate(atoms):
            uatom = UniqueAtom(atom)
            atom2word[uatom] = (str(atom), i)
        result = _generate_tok_pos(atom2word, edge)
        assert result.startswith('(')
        assert result.endswith(')')
