import re
import traceback

import spacy

import hyperbase.constants as const
from hyperbase.hyperedge import build_atom, hedge, non_unique, unique, UniqueAtom
from hyperbase.parsers import Parser

from hyperbase_parser_ab.alpha import Alpha
from hyperbase_parser_ab.lang_models import get_spacy_models
from hyperbase_parser_ab.rules import apply_rule, strict_rules, repair_rules


def _edge2txt_parts(edge, parse):
    atoms = [UniqueAtom(atom) for atom in edge.all_atoms()]
    tokens = [parse['atom2token'][atom] for atom in atoms if atom in parse['atom2token']]
    txts = [token.text for token in tokens]
    pos = [token.i for token in tokens]
    return list(zip(txts, txts, pos))


def _edge2text(edge, parse):
    if edge.not_atom and str(edge[0]) == const.possessive_builder:
        return _poss2text(edge, parse)

    parts = _edge2txt_parts(edge, parse)
    parts = sorted(parts, key=lambda x: x[2])

    prev_txt = None
    txt_parts = []
    sentence = str(parse['spacy_sentence'])
    for txt, _txt, _ in parts:
        if prev_txt is not None:
            res = re.search(r'{}(.*?){}'.format(re.escape(prev_txt), re.escape(txt)), sentence)
            if res:
                sep = res.group(1)
            else:
                sep = ' '
            if any(letter.isalnum() for letter in sep):
                sep = ' '
            txt_parts.append(sep)
        txt_parts.append(_txt)
        prev_txt = txt
    return ''.join(txt_parts)


def _concept_type_and_subtype(token):
    pos = token.pos_
    dep = token.dep_
    if dep == 'nmod':
        return 'Cm'
    if pos == 'ADJ':
        return 'Ca'
    elif pos == 'NOUN':
        return 'Cc'
    elif pos == 'PROPN':
        return 'Cp'
    elif pos == 'NUM':
        return 'C#'
    elif pos == 'DET':
        return 'Cd'
    elif pos == 'PRON':
        return 'Ci'
    else:
        return 'C'


def _modifier_type_and_subtype(token):
    pos = token.pos_
    dep = token.dep_
    if dep in {'neg', 'nk'}:
        return 'Mn'
    elif dep in {'poss', 'pg', 'ag'}:
        return 'Mp'
    elif dep == 'prep':
        return 'Mt'  # preposition
    elif dep == 'preconj':
        return 'Mj'  # conjunctional
    elif pos == 'ADJ':
        return 'Ma'
    elif pos == 'DET':
        return 'Md'
    elif pos == 'NUM':
        return 'M#'
    elif pos == 'AUX':
        return 'Mm'  # modal
    elif token.dep_ == 'prt':
        return 'Ml'  # particle
    elif pos == 'PART':
        return 'Mi'  # infinitive
    elif pos == 'ADV':  # adverb
        return 'M'  # quintissential modifier, no subtype needed
    else:
        return 'M'


def _builder_type_and_subtype(token):
    pos = token.pos_
    dep = token.dep_
    if dep in {'case', 'pg', 'ag'}:
        # if token.head.dep_ == 'poss':
        return 'Bp'
    elif pos == 'ADP':
        return 'Br'  # relational (proposition)
    elif pos == 'DET':
        return 'Bd'
    else:
        return 'B'


def _predicate_type_and_subtype(token):
    dep = token.dep_
    if dep in {'advcl', 'csubj', 'csubjpass', 'parataxis'}:
        return 'Pd'
    elif dep in {'relcl', 'ccomp', 'acl', 'pcomp', 'xcomp', 'rc'}:
        return 'P'
    elif _is_verb(token):
        return 'Pd'
    else:
        return 'P'


def _predicate_post_type_and_subtype(edge, subparts, args_string):
    return subparts[0]


def _is_verb(token):
    return token.pos_ == 'VERB'


def _poss2text(edge, parse):
    part1 = _edge2text(edge[1], parse).strip()
    part2 = _edge2text(edge[2], parse)
    if part1[-1] == 's':
        poss = "'"
    else:
        poss = "'s"
    return f'{part1}{poss} {part2}'


def _generate_tok_pos(atom2word, edge):
    if edge.atom:
        atom = unique(edge)
        if atom in atom2word:
            return str(atom2word[atom][1])
        else:
            return '-1'
    else:
        return '({})'.format(' '.join([_generate_tok_pos(atom2word, subedge) for subedge in edge]))


class AlphaBetaParser(Parser):
    def __init__(self, lang, beta='repair', normalize=True, post_process=True, debug=False):
        super().__init__()
        
        self.lang = lang

        models = get_spacy_models(lang)

        if len(models) == 0:
            raise RuntimeError(f"Language code '{lang}' is not recognized / language is nor supported.")        

        self.nlp = None
        for model in models:
            if spacy.util.is_package(model):
                self.nlp = spacy.load(model)
                print('Using language model: {}'.format(model))
                break
        if self.nlp is None:
            models_list = ", ".join(models)
            raise RuntimeError(f"Language '{lang}' requires one of the following language models to be installed:\n"
                               f"{models_list}.")
        
        self.alpha = Alpha(use_atomizer=True)

        if beta == 'strict':
            self.rules = strict_rules
        elif beta == 'repair':
            self.rules = repair_rules
        else:
            raise RuntimeError('unkown beta stage: {}'.format(beta))
        self.normalize = normalize
        self.post_process = post_process
        self.debug = debug

        self.atom2token = {}
        self.temp_atoms = set()
        self.orig_atom = {}
        self.token2atom = {}
        self.depths = {}
        self.connections = set()
        self.edge2text = {}
        self.edge2toks = {}
        self.toks2edge = {}
        self.cur_text = None
        self.doc = None
        self.beta = beta

    def debug_msg(self, msg):
        if self.debug:
            print(msg)
    
    def parse_sentence(self, sentence):
        # This runs spacy own sentensizer anyway...

        sentence = re.sub(r'\s+', ' ', sentence).strip()

        if self.nlp:
            self.reset(sentence)
            parses = []
            try:
                self.doc = self.nlp(sentence)
                offset = 0
                for sent in self.doc.sents:
                    parse = self.parse_spacy_sentence(sent, offset=offset)
                    if parse:
                        parses.append(parse)
                    offset += len(sent)
            except RuntimeError as error:
                print(error)
            return parses
        else:
            raise RuntimeError("spaCy model failed to initialize.")

    def parse_spacy_sentence(self, sent, atom_sequence=None, offset=0):
        try:
            if atom_sequence is None:
                atom_sequence = self._build_atom_sequence(sent)

            self._compute_depths_and_connections(sent.root)

            edge = None
            result, failed = self._parse_atom_sequence(atom_sequence)
            if result and len(result) == 1:
                edge = non_unique(result[0])
                # break

            if edge:
                edge = self._apply_arg_roles(edge)
                if self.beta == 'repair':
                    edge = self._repair(edge)
                if self.normalize:
                    edge = self._normalize(edge)
                if self.post_process:
                    edge = self._post_process(edge)
                atom2word = self._generate_atom2word(edge, offset=offset)
            
            if edge is None:
                return None
            
            return {
                'edge': edge,
                'failed': failed,
                'text': str(sent).strip(),
                'tokens': [str(token) for token in sent],
                'tok_pos': _generate_tok_pos(atom2word, edge)
            }
        except Exception as e:
            print('Caught exception: {} while parsing: "{}"'.format(str(e), str(sent)))
            traceback.print_exc()

    def manual_atom_sequence(self, sentence, token2atom):
        self.token2atom = {}

        atomseq = []
        for token in sentence:
            if token in token2atom:
                atom = token2atom[token]
            else:
                atom = None
            if atom:
                uatom = UniqueAtom(atom)
                self.dep_[uatom] = token
                self.token2atom[token] = uatom
                self.orig_atom[uatom] = uatom
                atomseq.append(uatom)
        return atomseq

    def reset(self, text):
        self.dep_ = {}
        self.temp_atoms = set()
        self.orig_atom = {}
        self.edge2toks = {}
        self.toks2edge = {}
        self.edge2coref = {}
        self.resolved_corefs = set()
        self.cur_text = text

    def _builder_arg_roles(self, edge):
        depth1 = self._dep_depth(edge[1])
        depth2 = self._dep_depth(edge[2])
        if depth1 < depth2:
            return 'ma'
        elif depth1 > depth2:
            return 'am'
        else:
            return 'mm'


    def _relation_arg_role(self, edge):
        head_token = self._head_token(edge)
        if not head_token:
            return '?'
        dep = head_token.dep_

        # subject
        if dep in {'nsubj', 'sb'}:
            return 's'
        # passive subject
        elif dep in {'nsubjpass', 'nsubj:pass'}:
            return 'p'
        # agent
        elif dep == 'agent':
            return 'a'
        # object
        elif dep in {'obj', 'dobj', 'pobj', 'prt', 'oprd', 'acomp', 'attr', 'ROOT', 'oa', 'pd'}:
            return 'o'
        # indirect object
        elif dep in {'iobj', 'dative', 'obl:arg', 'da'}:
            return 'i'
        # specifier
        elif dep in {'advcl', 'prep', 'npadvmod', 'advmod', 'mo', 'mnr'}:
            return 'x'
        # parataxis
        elif dep in {'parataxis', 'par'}:
            return 't'
        # interjection
        elif dep in {'intj', 'ng', 'dm'}:
            return 'j'
        # clausal complement
        elif dep in {'xcomp', 'ccomp', 'oc'}:
            return 'r'
        else:
            return '?'


    def _adjust_score(self, edges):
        min_depth = 9999999
        appos = False
        min_appos_depth = 9999999

        if all([edge.mtype() == 'C' for edge in edges]):
            for edge in edges:
                token = self._head_token(edge)
                depth = self.depths[self.token2atom[token]]
                if depth < min_depth:
                    min_depth = depth
                if token and token.dep_ == 'appos':
                    appos = True
                    if depth < min_appos_depth:
                        min_appos_depth = depth

        if appos and min_appos_depth > min_depth:
            return -99
        else:
            return 0

    def _head_token(self, edge):
        atoms = [unique(atom) for atom in edge.all_atoms() if unique(atom) in self.atom2token]
        min_depth = 9999999
        main_atom = None
        for atom in atoms:
            if atom in self.orig_atom:
                oatom = self.orig_atom[atom]
                if oatom in self.depths:
                    depth = self.depths[oatom]
                    if depth < min_depth:
                        min_depth = depth
                        main_atom = atom
        if main_atom:
            return self.atom2token[main_atom]
        else:
            return None
        
    def _dep_depth(self, edge):
        atoms = [unique(atom) for atom in edge.all_atoms() if unique(atom) in self.atom2token]
        mdepth = 99999999
        for atom in atoms:
            if atom in self.orig_atom:
                oatom = self.orig_atom[atom]
                if oatom in self.depths:
                    depth = self.depths[oatom]
                    if depth < mdepth:
                        mdepth = depth
        return mdepth

    def _build_atom(self, token, ent_type, last_token):
        text = token.text.lower()
        et = ent_type

        if ent_type[0] == 'P':
            atom = self._build_atom_predicate(token, ent_type, last_token)
        elif ent_type[0] == 'T':
            atom = self._build_atom_trigger(token, ent_type)
        elif ent_type[0] == 'M':
            atom = self._build_atom_modifier(token)
        else:
            atom = build_atom(text, et, self.lang)
        return atom

    def _build_atom_predicate(self, token, ent_type, last_token):
        text = token.text.lower()
        et = ent_type

        # first naive assignment of predicate subtype
        # (can be revised at post-processing stage)
        if ent_type == 'Pd':
            # interrogative cases
            if (last_token and
                    last_token.tag_ == '.' and
                    last_token.dep_ == 'punct' and
                    last_token.lemma_.strip() == '?'):
                ent_type = 'P?'
            # declarative (by default)
            else:
                ent_type = 'Pd'

        return build_atom(text, ent_type, self.lang)

    def _build_atom_trigger(self, token, ent_type):
        text = token.text.lower()
        et = 'Tv' if _is_verb(token) else ent_type
        return build_atom(text, et, self.lang)

    def _build_atom_modifier(self, token):
        text = token.text.lower()
        et = 'Mv' if _is_verb(token) else _modifier_type_and_subtype(token)
        return build_atom(text, et, self.lang)

    def _repair(self, edge):
        if edge.not_atom:
            edge = hedge([self._repair(subedge) for subedge in edge])

            if edge and len(edge) == 3 and str(edge[0])[:4] == '+/B.':
                if len(edge[1]) == 2 and edge[1].cmt == 'J':
                    return hedge([edge[1][0], edge[1][1], edge[2]])
                elif len(edge[2]) == 2 and edge[2].cmt == 'J':
                    return hedge([edge[2][0], edge[1], edge[2][1]])

        return edge

    def _normalize(self, edge):
        if edge.not_atom:
            edge = hedge([self._normalize(subedge) for subedge in edge])

            # Move modifier to internal connector if it is applied to
            # relations, specifiers or conjunctions
            if edge and edge.cmt == 'M' and not edge[1].atom:
                innner_conn = edge[1].cmt
                if innner_conn in {'P', 'T', 'J'}:
                    return hedge(((edge[0], edge[1][0]),) + edge[1][1:])

        return edge

    def _update_atom(self, old, new):
        uold = UniqueAtom(old)
        unew = UniqueAtom(new)
        if uold in self.atom2token:
            self.atom2token[unew] = self.atom2token[uold]
            self.temp_atoms.add(uold)
        self.orig_atom[unew] = uold

    def _replace_atom(self, edge, old, new):
        self._update_atom(old, new)
        return edge.replace_atom(old, new)

    def _insert_edge_with_argrole(self, edge, arg, argrole, pos):
        new_edge = edge.insert_edge_with_argrole(arg, argrole, pos)
        old_pred = edge[0].inner_atom()
        new_pred = new_edge[0].inner_atom()
        self._update_atom(old_pred, new_pred)
        return new_edge

    def _replace_argroles(self, edge, argroles):
        new_edge = edge.replace_argroles(argroles)
        old_pred = edge[0].inner_atom()
        new_pred = new_edge[0].inner_atom()
        self._update_atom(old_pred, new_pred)
        return new_edge

    def _apply_arg_roles(self, edge):
        if edge.atom:
            return edge

        new_entity = edge

        # Extend predicate connectors with argument types
        if edge.connector_mtype() == 'P':
            pred = edge.atom_with_type('P')
            subparts = pred.parts()[1].split('.')
            args = [self._relation_arg_role(param) for param in edge[1:]]
            args_string = ''.join(args)
            # TODO: this is done to detect imperative, to refactor
            pt = _predicate_post_type_and_subtype(edge, subparts, args_string)
            if len(subparts) > 2:
                new_part = '{}.{}.{}'.format(pt, args_string, subparts[2])
            else:
                new_part = '{}.{}'.format(pt, args_string)
            new_pred = pred.replace_atom_part(1, new_part)
            unew_pred = UniqueAtom(new_pred)
            upred = UniqueAtom(pred)
            self.atom2token[unew_pred] = self.atom2token[upred]
            self.temp_atoms.add(upred)
            self.orig_atom[unew_pred] = upred
            new_entity = edge.replace_atom(pred, new_pred, unique=True)

        # Extend builder connectors with argument types
        elif edge.connector_mtype() == 'B':
            builder = edge.atom_with_type('B')
            subparts = builder.parts()[1].split('.')
            arg_roles = self._builder_arg_roles(edge)
            if len(arg_roles) > 0:
                if len(subparts) > 1:
                    subparts[1] = arg_roles
                else:
                    subparts.append(arg_roles)
                new_part = '.'.join(subparts)
                new_builder = builder.replace_atom_part(1, new_part)
                ubuilder = UniqueAtom(builder)
                unew_builder = UniqueAtom(new_builder)
                if ubuilder in self.atom2token:
                    self.atom2token[unew_builder] = self.atom2token[ubuilder]
                    self.temp_atoms.add(ubuilder)
                self.orig_atom[unew_builder] = ubuilder
                new_entity = edge.replace_atom(builder, new_builder, unique=True)

        new_args = [self._apply_arg_roles(subentity) for subentity in new_entity[1:]]
        new_entity = hedge([new_entity[0]] + new_args)

        return new_entity

    def _generate_atom2word(self, edge, offset=0):
        atom2word = {}
        atoms = edge.all_atoms()
        for atom in atoms:
            uatom = UniqueAtom(atom)
            if uatom in self.atom2token:
                token = self.atom2token[uatom]
                word = (token.text, token.i - offset)
                atom2word[uatom] = word
        return atom2word

    def _parse_token(self, token, atom_type):
        if atom_type == 'X':
            return None
        elif atom_type == 'C':
            atom_type = _concept_type_and_subtype(token)
        elif atom_type == 'M':
            atom_type = _modifier_type_and_subtype(token)
        elif atom_type == 'B':
            atom_type = _builder_type_and_subtype(token)
        elif atom_type == 'P':
            atom_type = _predicate_type_and_subtype(token)

        # last token is useful to determine predicate subtype
        tokens = list(token.lefts) + list(token.rights)
        last_token = tokens[-1] if len(tokens) > 0 else None

        atom = self._build_atom(token, atom_type, last_token)
        self.debug_msg('ATOM: {}'.format(atom))

        return atom

    def _build_atom_sequence(self, sentence):
        features = []
        for pos, token in enumerate(sentence):
            head = token.head
            tag = token.tag_
            dep = token.dep_
            hpos = head.pos_ if head else ''
            hdep = head.dep_ if head else ''
            if pos + 1 < len(sentence):
                pos_after = sentence[pos + 1].pos_
            else:
                pos_after = ''
            features.append((tag, dep, hpos, hdep, pos_after))

        assert self.alpha is not None, "Alpha classifier must be initialized before parsing"
        atom_types = self.alpha.predict(sentence, features)

        self.token2atom = {}

        atomseq = []
        for token, atom_type in zip(sentence, atom_types):
            atom = self._parse_token(token, atom_type)
            if atom:
                uatom = UniqueAtom(atom)
                self.atom2token[uatom] = token
                self.token2atom[token] = uatom
                self.orig_atom[uatom] = uatom
                atomseq.append(uatom)
        return atomseq

    def _compute_depths_and_connections(self, root, depth=0):
        if depth == 0:
            self.depths = {}
            self.connections = set()

        if root in self.token2atom:
            parent_atom = self.token2atom[root]
            self.depths[parent_atom] = depth
        else:
            parent_atom = None

        for child in root.children:
            if parent_atom and child in self.token2atom:
                child_atom = self.token2atom[child]
                self.connections.add((parent_atom, child_atom))
                self.connections.add((child_atom, parent_atom))
            self._compute_depths_and_connections(child, depth + 1)

    def _is_pair_connected(self, atoms1, atoms2):
        for atom1 in atoms1:
            for atom2 in atoms2:
                if atom1 in self.orig_atom and atom2 in self.orig_atom:
                    pair = (self.orig_atom[atom1], self.orig_atom[atom2])
                    if pair in self.connections:
                        return True
        return False

    def _are_connected(self, atom_sets, connector_pos):
        conn = True
        for pos, arg in enumerate(atom_sets):
            if pos != connector_pos:
                if not self._is_pair_connected(atom_sets[connector_pos], arg):
                    conn = False
                    break
        return conn

    def _score(self, edges):
        atom_sets = [edge.all_atoms() for edge in edges]

        conn = False
        for pos in range(len(edges)):
            if self._are_connected(atom_sets, pos):
                conn = True
                break

        mdepth = 99999999
        for atom_set in atom_sets:
            for atom in atom_set:
                if atom in self.orig_atom:
                    oatom = self.orig_atom[atom]
                    if oatom in self.depths:
                        depth = self.depths[oatom]
                        if depth < mdepth:
                            mdepth = depth

        return (10000000 if conn else 0) + (mdepth * 100) + self._adjust_score(edges)

    def _parse_atom_sequence(self, atom_sequence):
        sequence = atom_sequence
        while True:
            action = None
            best_score = -999999999
            for rule_number, rule in enumerate(self.rules):
                window_start = rule.size - 1
                for pos in range(window_start, len(sequence)):
                    new_edge = apply_rule(rule, sequence, pos)
                    if new_edge:
                        score = self._score(sequence[pos - window_start:pos + 1])
                        score -= rule_number
                        if score > best_score:
                            action = (rule, score, new_edge, window_start, pos)
                            best_score = score

            # parse failed, make best effort to return something
            if action is None:
                # if all else fails...
                if len(sequence) > 0:
                    new_sequence = [hedge([':/J/.'] + sequence[:2])]
                    new_sequence += sequence[2:]
                else:
                    return None, True
            else:
                rule, s, new_edge, window_start, pos = action
                new_sequence = (sequence[:pos - window_start] + [new_edge] + sequence[pos + 1:])

                self.debug_msg('rule: {}'.format(rule))
                self.debug_msg('score: {}'.format(score))
                self.debug_msg('new_edge: {}'.format(new_edge))
                self.debug_msg('new_sequence: {}'.format(new_sequence))

            sequence = new_sequence
            if len(sequence) < 2:
                return sequence, False

    def sentensize(self, text):
        if self.nlp:
            doc = self.nlp(text.strip())
            return [str(sent).strip() for sent in doc.sents]
        else:
            raise RuntimeError("spaCy model failed to initialize.")

    def _edge2toks(self, edge):
        uatoms = [unique(atom) for atom in edge.all_atoms()]
        toks = tuple(sorted([self.atom2token[uatom] for uatom in uatoms if uatom in self.atom2token]))
        self.edge2toks[edge] = toks
        self.toks2edge[toks] = edge
        if edge.not_atom:
            for subedge in edge:
                self._edge2toks(subedge)

    # ===============
    # Post-processing
    # ===============
    def _insert_arg_in_tail(self, edge, arg):
        if edge.atom:
            return edge

        if edge.cmt == 'P':
            ars = edge.argroles()
            ar = None
            if 'p' in ars:
                if 'a' not in ars:
                    ar = 'a'
            elif 'a' in ars:
                ar = 'p'
            elif 's' not in ars:
                ar = 's'
            elif 'o' not in ars:
                ar = 'o'
            if ar:
                return self._insert_edge_with_argrole(edge, arg, ar, len(edge))

        new_tail = self._insert_arg_in_tail(edge[-1], arg)
        if new_tail != edge[-1]:
            return hedge(list(edge[:-1]) + [new_tail])
        if edge.cmt != 'P':
            return edge
        ars = edge.argroles()
        if ars == '':
            return edge
        return self._insert_edge_with_argrole(edge, arg, 'x', len(edge))

    def _insert_spec_rightmost_relation(self, edge, arg):
        if edge.atom:
            return edge
        if 'P' in [atom.mt for atom in edge[-1].atoms()]:
            return hedge(list(edge[:-1]) + [self._insert_spec_rightmost_relation(edge[-1], arg)])
        if edge[0].mt == 'P':
            return self._insert_edge_with_argrole(edge, arg, 'x', len(edge))
        for pos, subedge in reversed(list(enumerate(edge))):
            if 'P' in [atom.mt for atom in subedge.atoms()]:
                new_edge = list(edge)
                new_edge[pos] = self._insert_spec_rightmost_relation(subedge, arg)
                return hedge(new_edge)
        return edge

    def _process_colon_conjunctions(self, edge):
        if edge.atom:
            return edge
        edge = hedge([self._process_colon_conjunctions(subedge) for subedge in edge])
        if edge and str(edge[0]) == ':/J/.' and any(subedge.mt == 'R' for subedge in edge):
            if edge[1].mt == 'R':
                # RR
                if edge[2].mt == 'S':
                    # second is specification
                    return self._insert_edge_with_argrole(edge[1], edge[2], 'x', len(edge[1]))
                # RC
                elif edge[2].mt == 'C':
                    return self._insert_arg_in_tail(edge[1], edge[2])
            # CR
            elif edge[1].mt == 'C':
                if edge[2].mt == 'R':
                    if not 's' in edge[2].argroles():
                        # concept is subject
                        return self._insert_edge_with_argrole(edge[2], edge[1], 's', 0)
            # SR
            elif edge[1].mt == 'S':
                if edge[2].mt == 'R':
                    # first is specification
                    return self._insert_edge_with_argrole(edge[2], edge[1], 'x', len(edge[2]))
        return edge

    def _fix_argroles(self, edge):
        if edge.atom:
            return edge
        edge = hedge([self._fix_argroles(subedge) for subedge in edge])
        if edge is None:
            return edge
        ars = edge.argroles()
        if ars != '' and edge.mt == 'R':
            _ars = ''
            for ar, subedge in zip(ars, edge[1:]):
                _ar = ar
                if ar == '?':
                    if subedge.mt == 'R':
                        _ar = 'r'
                    elif subedge.mt == 'S':
                        _ar = 'x'
                _ars += _ar
            return self._replace_argroles(edge, _ars)
        return edge

    def _post_process(self, edge):
        if edge is None:
            return None
        _edge = self._fix_argroles(edge)
        _edge = self._process_colon_conjunctions(_edge)
        return _edge
