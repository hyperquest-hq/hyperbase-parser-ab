import re
import traceback
from typing import Any, cast

import hyperbase.constants as const
import spacy
from hyperbase.builders import build_atom, hedge
from hyperbase.hyperedge import (
    Atom,
    Hyperedge,
    UniqueAtom,
    non_unique,
    unique,
)
from hyperbase.parsers import Parser, ParseResult
from hyperbase.parsers.utils import edge_depth_exceeds
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from hyperbase_parser_ab.alpha import Alpha
from hyperbase_parser_ab.lang_models import SPACY_MODELS
from hyperbase_parser_ab.rules import RULES, Rule, apply_rule


def _edge2txt_parts(
    edge: Hyperedge, parse: dict[str, Any]
) -> list[tuple[str, str, int]]:
    atoms: list[Atom] = [UniqueAtom(atom) for atom in edge.all_atoms()]
    tokens: list[Token] = [
        parse["atom2token"][atom] for atom in atoms if atom in parse["atom2token"]
    ]
    txts: list[str] = [token.text for token in tokens]
    pos: list[int] = [token.i for token in tokens]
    return list(zip(txts, txts, pos, strict=True))


def _edge2text(edge: Hyperedge, parse: dict[str, Any]) -> str:
    if edge.not_atom and str(edge[0]) == const.possessive_builder:
        return _poss2text(edge, parse)

    parts: list[tuple[str, str, int]] = _edge2txt_parts(edge, parse)
    parts = sorted(parts, key=lambda x: x[2])

    prev_txt: str | None = None
    txt_parts: list[str] = []
    sentence: str = str(parse["spacy_sentence"])
    for txt, _txt, _ in parts:
        if prev_txt is not None:
            res: re.Match[str] | None = re.search(
                rf"{re.escape(prev_txt)}(.*?){re.escape(txt)}", sentence
            )
            if res:
                sep: str = res.group(1)
            else:
                sep = " "
            if any(letter.isalnum() for letter in sep):
                sep = " "
            txt_parts.append(sep)
        txt_parts.append(_txt)
        prev_txt = txt
    return "".join(txt_parts)


# fix known cases where classifier fails
def _fix_atom_type(atom_type: str, token: Token) -> str:
    tag: str = token.tag_
    dep: str = token.dep_
    if tag == "ADP" and dep == "case":
        return "B"
    else:
        return atom_type


def _concept_type_and_subtype(token: Token) -> str:
    pos: str = token.pos_
    dep: str = token.dep_
    if dep == "nmod":
        return "Cm"
    return {
        "ADJ": "Ca",
        "NOUN": "Cc",
        "PROPN": "Cp",
        "NUM": "C#",
        "DET": "Cd",
        "PRON": "Ci",
    }.get(pos, "C")


def _modifier_type_and_subtype(token: Token) -> str:
    pos: str = token.pos_
    dep: str = token.dep_
    if dep in {"neg", "nk"}:
        return "Mn"
    elif dep in {"poss", "pg", "ag"}:
        return "Mp"
    elif dep == "prep":
        return "Mt"  # preposition
    elif dep == "preconj":
        return "Mj"  # conjunctional
    elif pos == "ADJ":
        return "Ma"
    elif pos == "DET":
        return "Md"
    elif pos == "NUM":
        return "M#"
    elif pos == "AUX":
        return "Mm"  # modal
    elif token.dep_ == "prt":
        return "Ml"  # particle
    elif pos == "PART":
        return "Mi"  # infinitive
    elif pos == "ADV":  # adverb
        return "Mb"
    else:
        return "M"


def _builder_type_and_subtype(token: Token) -> str:
    pos: str = token.pos_
    dep: str = token.dep_
    if dep in {"case", "pg", "ag"}:
        return "Bp"
    elif pos == "ADP":
        return "Br"  # relational (proposition)
    elif pos == "DET":
        return "Bd"
    else:
        return "B"


def _predicate_type_and_subtype(token: Token) -> str:
    dep: str = token.dep_
    if dep in {"advcl", "csubj", "csubjpass", "parataxis"}:
        return "Pd"
    elif dep in {"relcl", "ccomp", "acl", "pcomp", "xcomp", "rc"}:
        return "P"
    elif _is_verb(token):
        return "Pd"
    else:
        return "P"


def _predicate_post_type_and_subtype(
    edge: Hyperedge, subparts: list[str], args_string: str
) -> str:
    return subparts[0]


def _is_verb(token: Token) -> bool:
    return token.pos_ == "VERB"


def _poss2text(edge: Hyperedge, parse: dict[str, Any]) -> str:
    part1: str = _edge2text(edge[1], parse).strip()
    part2: str = _edge2text(edge[2], parse)
    if part1[-1] == "s":
        poss: str = "'"
    else:
        poss = "'s"
    return f"{part1}{poss} {part2}"


def _generate_tok_pos(atom2word: dict[Atom, tuple[str, int]], edge: Hyperedge) -> str:
    if edge.atom:
        uatom: Atom = cast(Atom, unique(edge))
        if uatom is not None and uatom in atom2word:
            return str(atom2word[uatom][1])
        else:
            return "-1"
    else:
        return "({})".format(
            " ".join([_generate_tok_pos(atom2word, subedge) for subedge in edge])
        )


class AlphaBetaParser(Parser):
    @classmethod
    def accepted_params(cls) -> dict[str, dict[str, Any]]:
        return {
            **super().accepted_params(),
            "lang": {
                "type": str,
                "default": None,
                "description": "Language code (e.g. 'de', 'en', 'fr').",
                "required": True,
            },
            "debug": {
                "type": bool,
                "default": False,
                "description": "Enable debug message output.",
                "required": False,
            },
            "lang_namespace": {
                "type": bool,
                "default": False,
                "description": (
                    "Include the language code as a namespace in atoms "
                    "(e.g. 'apple/Cc/en' instead of 'apple/Cc')."
                ),
                "required": False,
            },
        }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        self.lang: str = self.params["lang"]

        if self.lang not in SPACY_MODELS:
            raise RuntimeError(f"Language code '{self.lang}' is not recognized.")

        debug: bool = self.params.get("debug", False)
        lang_namespace: bool = self.params.get("lang_namespace", False)
        self.atom_lang: str = self.lang if lang_namespace else ""

        models: list[str] = SPACY_MODELS[self.lang]

        self.nlp: Language | None = None
        for model in models:
            if spacy.util.is_package(model):
                self.nlp = spacy.load(model)
                print(f"Using language model: {model}")
                break
        if self.nlp is None:
            models_list: str = ", ".join(models)
            raise RuntimeError(
                f"Language '{self.lang}' requires one of the following "
                f"language models:\n{models_list}."
            )

        self.alpha: Alpha = Alpha(use_atomizer=True)

        self.debug: bool = debug

        self.atom2token: dict[Atom, Token] = {}
        self.temp_atoms: set[Atom] = set()
        self.orig_atom: dict[Atom, UniqueAtom] = {}
        self.token2atom: dict[Token, Atom] = {}
        self.depths: dict[Atom, int] = {}
        self.connections: set[tuple[Atom, Atom]] = set()
        self.edge2text: dict[Hyperedge, str] = {}
        self.edge2toks: dict[Hyperedge, tuple[Token, ...]] = {}
        self.toks2edge: dict[tuple[Token, ...], Hyperedge] = {}
        self.cur_text: str | None = None
        self.doc: Doc | None = None

    def debug_msg(self, msg: str) -> None:
        if self.debug:
            print(msg)

    def install_repl(self, session: object) -> None:
        from hyperbase_parser_ab.repl import install

        install(self, session)

    def parse_sentence(self, sentence: str) -> list[ParseResult]:
        # This runs spacy own sentensizer anyway...

        sentence = re.sub(r"\s+", " ", sentence).strip()

        if self.nlp:
            self.reset(sentence)
            parses: list[ParseResult] = []
            try:
                self.doc = self.nlp(sentence)
                offset: int = 0
                for sent in self.doc.sents:
                    parse: ParseResult | None = self.parse_spacy_sentence(
                        sent, offset=offset
                    )
                    if parse:
                        parses.append(parse)
                    offset += len(sent)
            except RuntimeError as error:
                print(error)
            return parses
        else:
            raise RuntimeError("spaCy model failed to initialize.")

    def parse_spacy_sentence(
        self, sent: Span, atom_sequence: list[Atom] | None = None, offset: int = 0
    ) -> ParseResult | None:
        try:
            if atom_sequence is None:
                atom_sequence = self._build_atom_sequence(sent)

            self._compute_depths_and_connections(sent.root)

            edge: Hyperedge | None = None
            result: list[Hyperedge] | None
            failed: bool
            result, failed = self._parse_atom_sequence(atom_sequence)
            if result and len(result) == 1:
                edge = non_unique(result[0])
                self.debug_msg(f"Initial parse: {edge!s}")

            # Reject pathologically deep parses before they reach the
            # recursive transforms below (which would otherwise blow the
            # Python stack on inputs with extreme nesting).
            if edge is not None and edge_depth_exceeds(edge, self.max_depth):
                self.debug_msg(
                    f"Rejecting parse: edge depth exceeds max_depth="
                    f"{self.max_depth} for sentence: {sent!s}"
                )
                return None

            atom2word: dict[Atom, tuple[str, int]] = {}
            if edge:
                edge = self._apply_arg_roles(edge)
                self.debug_msg(f"After applying argument roles: {edge!s}")
                edge = self._repair(edge)
                self.debug_msg(f"After repair: {edge!s}")
                edge = self._normalise_modifiers(edge)
                self.debug_msg(f"After modifier normalisation: {edge!s}")
                edge = self._post_process(edge)
                self.debug_msg(f"After post-processing: {edge!s}")
                if edge is not None:
                    atom2word = self._generate_atom2word(edge, offset=offset)

            if edge is None:
                return None

            tok_pos_str = _generate_tok_pos(atom2word, edge)
            return ParseResult(
                edge=edge,
                text=str(sent).strip(),
                tokens=[str(token) for token in sent],
                tok_pos=hedge(tok_pos_str),
                failed=failed,
            )
        except Exception as e:
            print(f'Caught exception: {e!s} while parsing: "{sent!s}"')
            traceback.print_exc()
            return None

    def manual_atom_sequence(
        self, sentence: Span, token2atom: dict[Token, Atom]
    ) -> list[Atom]:
        self.token2atom = {}

        atomseq: list[Atom] = []
        for token in sentence:
            if token in token2atom:
                atom: Atom | None = token2atom[token]
            else:
                atom = None
            if atom:
                uatom: Atom = UniqueAtom(atom)
                self.dep_[uatom] = token
                self.token2atom[token] = uatom
                self.orig_atom[uatom] = uatom
                atomseq.append(uatom)
        return atomseq

    def reset(self, text: str) -> None:
        self.dep_: dict[Atom, Token] = {}
        self.temp_atoms = set()
        self.orig_atom = {}
        self.edge2toks = {}
        self.toks2edge = {}
        self.edge2coref: dict[Hyperedge, object] = {}
        self.resolved_corefs: set[object] = set()
        self.cur_text = text

    def _builder_arg_roles(self, edge: Hyperedge) -> str:
        depth1: int = self._dep_depth(edge[1])
        depth2: int = self._dep_depth(edge[2])
        if depth1 < depth2:
            return "ma"
        elif depth1 > depth2:
            return "am"
        else:
            return "mm"

    def _relation_arg_role(self, edge: Hyperedge) -> str:
        head_token: Token | None = self._head_token(edge)
        if not head_token:
            return "?"
        dep: str = head_token.dep_

        # subject
        if dep in {"nsubj", "sb"}:
            return "s"
        # passive subject (becomes object)
        elif dep in {"nsubjpass", "nsubj:pass"}:
            return "o"
        # agent (becomes subject)
        elif dep == "agent":
            return "s"
        # object
        elif dep in {
            "obj",
            "dobj",
            "pobj",
            "prt",
            "oprd",
            "acomp",
            "attr",
            "ROOT",
            "oa",
            "pd",
            # clausal complement, these are probably going to be nested relations
            "xcomp",
            "ccomp",
            "oc",
        }:
            return "o"
        # indirect object
        elif dep in {"iobj", "dative", "obl:arg", "da"} or dep in {
            "advcl",
            "prep",
            "npadvmod",
            "advmod",
            "mo",
            "mnr",
        }:
            return "x"
        else:
            return "?"

    def _adjust_score(self, edges: list[Hyperedge]) -> int:
        min_depth: int = 9999999
        appos: bool = False
        min_appos_depth: int = 9999999

        if all(edge.mtype() == "C" for edge in edges):
            for edge in edges:
                token: Token | None = self._head_token(edge)
                if token is None:
                    continue
                depth: int = self.depths[self.token2atom[token]]
                if depth < min_depth:
                    min_depth = depth
                if token.dep_ == "appos":
                    appos = True
                    if depth < min_appos_depth:
                        min_appos_depth = depth

        if appos and min_appos_depth > min_depth:
            return -99
        else:
            return 0

    def _head_token(self, edge: Hyperedge) -> Token | None:
        atoms: list[Atom] = [
            cast(Atom, uatom)
            for atom in edge.all_atoms()
            if (uatom := unique(atom)) is not None and uatom in self.atom2token
        ]
        min_depth: int = 9999999
        main_atom: Atom | None = None
        for atom in atoms:
            if atom in self.orig_atom:
                oatom: Atom = self.orig_atom[atom]
                if oatom in self.depths:
                    depth: int = self.depths[oatom]
                    if depth < min_depth:
                        min_depth = depth
                        main_atom = atom
        if main_atom:
            return self.atom2token[main_atom]
        else:
            return None

    def _dep_depth(self, edge: Hyperedge) -> int:
        atoms: list[Atom] = [
            cast(Atom, uatom)
            for atom in edge.all_atoms()
            if (uatom := unique(atom)) is not None and uatom in self.atom2token
        ]
        mdepth: int = 99999999
        for atom in atoms:
            if atom in self.orig_atom:
                oatom: Atom = self.orig_atom[atom]
                if oatom in self.depths:
                    depth: int = self.depths[oatom]
                    if depth < mdepth:
                        mdepth = depth
        return mdepth

    def _build_atom(
        self, token: Token, ent_type: str, last_token: Token | None
    ) -> Atom:
        text: str = token.text.lower()
        et: str = ent_type

        if ent_type[0] == "P":
            atom: Atom = self._build_atom_predicate(token, ent_type, last_token)
        elif ent_type[0] == "T":
            atom = self._build_atom_trigger(token, ent_type)
        elif ent_type[0] == "M":
            atom = self._build_atom_modifier(token)
        else:
            atom = build_atom(text, et, self.atom_lang)
        return atom

    def _build_atom_predicate(
        self, token: Token, ent_type: str, last_token: Token | None
    ) -> Atom:
        text: str = token.text.lower()

        # first naive assignment of predicate subtype
        # (can be revised at post-processing stage)
        if ent_type == "Pd":
            # interrogative cases
            if (
                last_token
                and last_token.tag_ == "."
                and last_token.dep_ == "punct"
                and last_token.lemma_.strip() == "?"
            ):
                ent_type = "P?"
            # declarative (by default)
            else:
                ent_type = "Pd"

        return build_atom(text, ent_type, self.atom_lang)

    def _build_atom_trigger(self, token: Token, ent_type: str) -> Atom:
        text: str = token.text.lower()

        # indirect object
        if token.dep_ in {"iobj", "dative", "obl:arg", "da"}:
            et = "Ti"
        elif _is_verb(token):
            et = "Tv"
        else:
            et = ent_type

        return build_atom(text, et, self.atom_lang)

    def _build_atom_modifier(self, token: Token) -> Atom:
        text: str = token.text.lower()
        et: str = "Mv" if _is_verb(token) else _modifier_type_and_subtype(token)
        return build_atom(text, et, self.atom_lang)

    def _repair(self, edge: Hyperedge) -> Hyperedge:
        if edge.not_atom:
            new_edge: Hyperedge = hedge([self._repair(subedge) for subedge in edge])
            if new_edge is None:
                return edge
            edge = new_edge

            if len(edge) == 3 and str(edge[0])[:4] == "+/B.":
                if len(edge[1]) == 2 and edge[1].cmt == "J":
                    return hedge([edge[1][0], edge[1][1], edge[2]])
                elif len(edge[2]) == 2 and edge[2].cmt == "J":
                    return hedge([edge[2][0], edge[1], edge[2][1]])

        return edge

    def _normalise_modifiers(self, edge: Hyperedge) -> Hyperedge:
        if edge.not_atom:
            new_edge: Hyperedge | None = hedge(
                [self._normalise_modifiers(subedge) for subedge in edge]
            )
            if new_edge is None:
                return edge
            edge = new_edge

            # Move modifier to internal connector if it is applied to
            # relations, specifiers or conjunctions
            if edge.cmt == "M" and not edge[1].atom:
                innner_conn: str | None = edge[1].cmt
                if innner_conn in {"P", "T"}:
                    return hedge(((edge[0], edge[1][0]), *edge[1][1:]))

        return edge

    def _update_atom(self, old: Atom, new: Atom) -> None:
        uold: Atom = UniqueAtom(old)
        unew: Atom = UniqueAtom(new)
        if uold in self.atom2token:
            self.atom2token[unew] = self.atom2token[uold]
            self.temp_atoms.add(uold)
        self.orig_atom[unew] = uold

    def _add_argument(
        self, edge: Hyperedge, arg: Hyperedge, argrole: str, pos: int
    ) -> Hyperedge:
        new_edge: Hyperedge = edge.add_argument(arg, argrole, pos)
        old_pred: Atom = edge[0].inner_atom()
        new_pred: Atom = new_edge[0].inner_atom()
        self._update_atom(old_pred, new_pred)
        return new_edge

    def _replace_argroles(self, edge: Hyperedge, argroles: str) -> Hyperedge:
        new_edge: Hyperedge = edge.replace_argroles(argroles)
        old_pred: Atom = edge[0].inner_atom()
        new_pred: Atom = new_edge[0].inner_atom()
        self._update_atom(old_pred, new_pred)
        return new_edge

    def _apply_arg_roles(self, edge: Hyperedge) -> Hyperedge:
        if edge.atom:
            return edge

        new_entity: Hyperedge = edge

        # Extend predicate connectors with argument types
        if edge.connector_mtype() == "P":
            pred: Atom | None = edge.atom_with_type("P")
            assert pred is not None
            subparts: list[str] = pred.parts()[1].split(".")
            args: list[str] = [self._relation_arg_role(param) for param in edge[1:]]
            args_string: str = "".join(args)
            # TODO: this is done to detect imperative, to refactor
            pt: str = _predicate_post_type_and_subtype(edge, subparts, args_string)
            if len(subparts) > 2:
                new_part: str = f"{pt}.{args_string}.{subparts[2]}"
            else:
                new_part = f"{pt}.{args_string}"
            new_pred: Atom = pred.replace_atom_part(1, new_part)
            unew_pred: Atom = UniqueAtom(new_pred)
            upred: Atom = UniqueAtom(pred)
            self.atom2token[unew_pred] = self.atom2token[upred]
            self.temp_atoms.add(upred)
            self.orig_atom[unew_pred] = upred
            new_entity = edge.replace_atom(pred, new_pred, unique=True)

        # Extend builder connectors with argument types
        elif edge.connector_mtype() == "B":
            builder: Atom | None = edge.atom_with_type("B")
            assert builder is not None
            subparts = builder.parts()[1].split(".")
            arg_roles: str = self._builder_arg_roles(edge)
            if len(arg_roles) > 0:
                if len(subparts) > 1:
                    subparts[1] = arg_roles
                else:
                    subparts.append(arg_roles)
                new_part = ".".join(subparts)
                new_builder: Atom = builder.replace_atom_part(1, new_part)
                ubuilder: Atom = UniqueAtom(builder)
                unew_builder: Atom = UniqueAtom(new_builder)
                if ubuilder in self.atom2token:
                    self.atom2token[unew_builder] = self.atom2token[ubuilder]
                    self.temp_atoms.add(ubuilder)
                self.orig_atom[unew_builder] = ubuilder
                new_entity = edge.replace_atom(builder, new_builder, unique=True)

        new_args: list[Hyperedge] = [
            self._apply_arg_roles(subentity) for subentity in new_entity[1:]
        ]
        new_entity = hedge([new_entity[0], *new_args])

        return new_entity

    def _generate_atom2word(
        self, edge: Hyperedge, offset: int = 0
    ) -> dict[Atom, tuple[str, int]]:
        atom2word: dict[Atom, tuple[str, int]] = {}
        atoms: list[Atom] = edge.all_atoms()
        for atom in atoms:
            uatom: Atom = UniqueAtom(atom)
            if uatom in self.atom2token:
                token: Token = self.atom2token[uatom]
                word: tuple[str, int] = (token.text, token.i - offset)
                atom2word[uatom] = word
        return atom2word

    def _parse_token(self, token: Token, atom_type: str) -> Atom | None:
        atom_type = _fix_atom_type(atom_type, token)
        if atom_type == "X":
            return None
        elif atom_type == "C":
            atom_type = _concept_type_and_subtype(token)
        elif atom_type == "M":
            atom_type = _modifier_type_and_subtype(token)
        elif atom_type == "B":
            atom_type = _builder_type_and_subtype(token)
        elif atom_type == "P":
            atom_type = _predicate_type_and_subtype(token)

        # last token is useful to determine predicate subtype
        tokens: list[Token] = list(token.lefts) + list(token.rights)
        last_token: Token | None = tokens[-1] if len(tokens) > 0 else None

        atom: Atom = self._build_atom(token, atom_type, last_token)
        self.debug_msg(f"ATOM: {atom}")

        return atom

    def _build_atom_sequence(self, sentence: Span) -> list[Atom]:
        features: list[tuple[str, str, str, str, str]] = []
        for pos, token in enumerate(sentence):
            head: Token = token.head
            tag: str = token.tag_
            dep: str = token.dep_
            hpos: str = head.pos_ if head else ""
            hdep: str = head.dep_ if head else ""
            if pos + 1 < len(sentence):
                pos_after: str = sentence[pos + 1].pos_
            else:
                pos_after = ""
            features.append((tag, dep, hpos, hdep, pos_after))

        assert self.alpha is not None, "Alpha must be initialized before parsing"
        atom_types: tuple[str, ...] | list[str] = self.alpha.predict(sentence, features)

        self.token2atom = {}

        atomseq: list[Atom] = []
        for token, atom_type in zip(sentence, atom_types, strict=True):
            atom: Atom | None = self._parse_token(token, atom_type)
            if atom:
                uatom: Atom = UniqueAtom(atom)
                self.atom2token[uatom] = token
                self.token2atom[token] = uatom
                self.orig_atom[uatom] = uatom
                atomseq.append(uatom)
        self.debug_msg(f"Atom sequence: {atomseq}")
        return atomseq

    def _compute_depths_and_connections(self, root: Token, depth: int = 0) -> None:
        if depth == 0:
            self.depths = {}
            self.connections = set()

        if root in self.token2atom:
            parent_atom: Atom | None = self.token2atom[root]
            self.depths[parent_atom] = depth
        else:
            parent_atom = None

        for child in root.children:
            if parent_atom and child in self.token2atom:
                child_atom: Atom = self.token2atom[child]
                self.connections.add((parent_atom, child_atom))
                self.connections.add((child_atom, parent_atom))
            self._compute_depths_and_connections(child, depth + 1)

    def _is_pair_connected(self, atoms1: list[Atom], atoms2: list[Atom]) -> bool:
        for atom1 in atoms1:
            for atom2 in atoms2:
                if atom1 in self.orig_atom and atom2 in self.orig_atom:
                    pair: tuple[Atom, Atom] = (
                        self.orig_atom[atom1],
                        self.orig_atom[atom2],
                    )
                    if pair in self.connections:
                        return True
        return False

    def _are_connected(self, atom_sets: list[list[Atom]], connector_pos: int) -> bool:
        conn: bool = True
        for pos, arg in enumerate(atom_sets):
            if pos != connector_pos and not self._is_pair_connected(
                atom_sets[connector_pos], arg
            ):
                conn = False
                break
        return conn

    def _score(self, edges: list[Hyperedge]) -> int:
        atom_sets: list[list[Atom]] = [edge.all_atoms() for edge in edges]

        conn: bool = False
        for pos in range(len(edges)):
            if self._are_connected(atom_sets, pos):
                conn = True
                break

        mdepth: int = 99999999
        for atom_set in atom_sets:
            for atom in atom_set:
                if atom in self.orig_atom:
                    oatom: Atom = self.orig_atom[atom]
                    if oatom in self.depths:
                        depth: int = self.depths[oatom]
                        if depth < mdepth:
                            mdepth = depth

        return (10000000 if conn else 0) + (mdepth * 100) + self._adjust_score(edges)

    def _parse_atom_sequence(
        self, atom_sequence: list[Atom]
    ) -> tuple[list[Hyperedge] | None, bool]:
        sequence: list[Hyperedge] = list(atom_sequence)
        while True:
            action: tuple[Rule, int, Hyperedge, int, int] | None = None
            best_score: int = -999999999
            for rule_number, rule in enumerate(RULES):
                window_start: int = rule.size - 1
                for pos in range(window_start, len(sequence)):
                    new_edge: Hyperedge | None = apply_rule(rule, sequence, pos)
                    if new_edge:
                        score: int = self._score(sequence[pos - window_start : pos + 1])
                        score -= rule_number
                        if score > best_score:
                            action = (rule, score, new_edge, window_start, pos)
                            best_score = score

            # parse failed, make best effort to return something
            if action is None:
                # if all else fails...
                if len(sequence) > 0:
                    fallback: Hyperedge = hedge([":/J/.", *sequence[:2]])
                    new_sequence: list[Hyperedge] = (
                        [fallback] if fallback else []
                    ) + sequence[2:]
                else:
                    return None, True
            else:
                rule, _, new_edge, window_start, pos = action
                new_sequence = [
                    *sequence[: pos - window_start],
                    new_edge,
                    *sequence[pos + 1 :],
                ]

                self.debug_msg(f"rule: {rule}")
                self.debug_msg(f"score: {score}")
                self.debug_msg(f"new_edge: {new_edge}")
                self.debug_msg(f"new_sequence: {new_sequence}")

            sequence = new_sequence
            if len(sequence) < 2:
                return sequence, False

    def get_sentences(self, text: str) -> list[str]:
        if self.nlp:
            doc: Doc = self.nlp(text.strip())
            return [str(sent).strip() for sent in doc.sents]
        else:
            raise RuntimeError("spaCy model failed to initialize.")

    def _edge2toks(self, edge: Hyperedge) -> None:
        uatoms: list[Hyperedge] = [unique(atom) for atom in edge.all_atoms()]
        toks: tuple[Token, ...] = tuple(
            sorted(
                [
                    self.atom2token[cast(Atom, uatom)]
                    for uatom in uatoms
                    if uatom is not None and uatom in self.atom2token
                ]
            )
        )
        self.edge2toks[edge] = toks
        self.toks2edge[toks] = edge
        if edge.not_atom:
            for subedge in edge:
                self._edge2toks(subedge)

    # ===============
    # Post-processing
    # ===============
    def _insert_arg_in_tail(self, edge: Hyperedge, arg: Hyperedge) -> Hyperedge:
        if edge.atom:
            return edge

        if edge.cmt == "P":
            ars: str = edge.argroles()
            ar: str | None = None
            if "p" in ars:
                if "a" not in ars:
                    ar = "a"
            elif "a" in ars:
                ar = "p"
            elif "s" not in ars:
                ar = "s"
            elif "o" not in ars:
                ar = "o"
            if ar:
                return self._add_argument(edge, arg, ar, len(edge))

        new_tail: Hyperedge = self._insert_arg_in_tail(edge[-1], arg)
        if new_tail != edge[-1]:
            return hedge([*list(edge[:-1]), new_tail])
        if edge.cmt != "P":
            return edge
        ars = edge.argroles()
        if ars == "":
            return edge
        return self._add_argument(edge, arg, "x", len(edge))

    def _insert_spec_rightmost_relation(
        self, edge: Hyperedge, arg: Hyperedge
    ) -> Hyperedge:
        if edge.atom:
            return edge
        if "P" in [atom.mt for atom in edge[-1].atoms()]:
            return hedge(
                [*edge[:-1], self._insert_spec_rightmost_relation(edge[-1], arg)]
            )
        if edge[0].mt == "P":
            return self._add_argument(edge, arg, "x", len(edge))
        for pos, subedge in reversed(list(enumerate(edge))):
            if "P" in [atom.mt for atom in subedge.atoms()]:
                new_edge_list: list[Hyperedge] = list(edge)
                new_edge_list[pos] = self._insert_spec_rightmost_relation(subedge, arg)
                return hedge(new_edge_list)
        return edge

    def _process_colon_conjunctions(self, edge: Hyperedge) -> Hyperedge:
        if edge.atom:
            return edge
        new_edge: Hyperedge = hedge(
            [self._process_colon_conjunctions(subedge) for subedge in edge]
        )
        if new_edge is None:
            return edge
        edge = new_edge
        if str(edge[0]) == ":/J/." and any(subedge.mt == "R" for subedge in edge):
            if edge[1].mt == "R":
                # RR
                if edge[2].mt == "S":
                    # second is specification
                    return self._add_argument(edge[1], edge[2], "x", len(edge[1]))
                # RC
                elif edge[2].mt == "C":
                    return self._insert_arg_in_tail(edge[1], edge[2])
            # CR
            elif edge[1].mt == "C":
                if edge[2].mt == "R" and "s" not in edge[2].argroles():
                    # concept is subject
                    return self._add_argument(edge[2], edge[1], "s", 0)
            # SR
            elif edge[1].mt == "S":  # noqa: SIM102
                if edge[2].mt == "R":
                    # first is specification
                    return self._add_argument(edge[2], edge[1], "x", len(edge[2]))
        return edge

    def _fix_argroles(self, edge: Hyperedge) -> Hyperedge:
        if edge.atom:
            return edge
        new_edge: Hyperedge = hedge([self._fix_argroles(subedge) for subedge in edge])
        if new_edge is None:
            return edge
        edge = new_edge
        ars: str = edge.argroles()
        if ars != "" and edge.mt == "R":
            _ars: str = ""
            for ar, subedge in zip(ars, edge[1:], strict=True):
                _ar: str = ar
                if ar == "?" and subedge.mt in {"R", "S"}:
                    _ar = "x"
                _ars += _ar
            return self._replace_argroles(edge, _ars)
        return edge

    def _fix_spec_object(self, edge: Hyperedge) -> Hyperedge:
        if edge.atom:
            return edge
        new_edge: Hyperedge = hedge(
            [self._fix_spec_object(subedge) for subedge in edge]
        )
        if new_edge is None:
            return edge
        edge = new_edge

        ars: str = edge.argroles()
        if edge.mt != "R" or "o" not in ars:
            return edge

        for i, ar in enumerate(ars):
            if ar == "o":
                arg: Hyperedge = edge[i + 1]
                if (
                    arg.not_atom
                    and arg.mt == "S"
                    and any(subedge.mt == "R" for subedge in arg[1:])
                ):
                    # Get trigger and change type from T to M
                    trigger: Hyperedge = arg[0]
                    trigger_atom: Atom = trigger.inner_atom()
                    new_mod_atom: Atom = trigger_atom.replace_atom_part(1, "Mr")
                    self._update_atom(trigger_atom, new_mod_atom)

                    if trigger.atom:
                        new_mod: Hyperedge = new_mod_atom
                    else:
                        new_mod = trigger.replace_atom(
                            trigger_atom, new_mod_atom, unique=True
                        )

                    # Wrap predicate connector with modifier
                    new_connector: Hyperedge = hedge((new_mod, edge[0]))

                    # Replace specification with its arguments
                    new_args: list[Hyperedge] = (
                        list(edge[1 : i + 1]) + list(arg[1:]) + list(edge[i + 2 :])
                    )
                    return hedge([new_connector, *new_args])

        return edge

    def _flatten_conjunctions(self, edge: Hyperedge) -> Hyperedge:
        if edge.atom:
            return edge
        new_edge: Hyperedge = hedge(
            [self._flatten_conjunctions(subedge) for subedge in edge]
        )
        if new_edge is None:
            return edge
        edge = new_edge
        if edge[0].mt != "J":
            return edge
        connector: Hyperedge = edge[0]
        flattened: list[Hyperedge] = [connector]
        changed: bool = False
        for subedge in edge[1:]:
            if subedge.not_atom and len(subedge) >= 2 and subedge[0] == connector:
                flattened.extend(list(subedge[1:]))
                changed = True
            else:
                flattened.append(subedge)
        if changed:
            return hedge(flattened)
        return edge

    def _post_process(self, edge: Hyperedge | None) -> Hyperedge | None:
        if edge is None:
            return None
        _edge: Hyperedge = self._fix_argroles(edge)
        _edge = self._fix_spec_object(_edge)
        _edge = self._process_colon_conjunctions(_edge)
        _edge = self._flatten_conjunctions(_edge)
        return _edge
