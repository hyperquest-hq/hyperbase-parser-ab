import re
import traceback
from dataclasses import dataclass, field
from typing import Any, cast

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
from hyperbase.parsers.badness import badness_check, check_structural_quality
from hyperbase.parsers.utils import edge_depth_exceeds
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from hyperbase_parser_ab.alpha import Alpha
from hyperbase_parser_ab.lang_models import SPACY_MODELS
from hyperbase_parser_ab.rules import RULES, apply_rule
from hyperbase_parser_ab.trace import (
    AtomTrace,
    ParseTrace,
    RuleCandidate,
    RuleIteration,
    rule_repr,
)


def _concept_type_and_subtype(token: Token) -> str:
    pos: str = token.pos_
    dep: str = token.dep_
    if dep == "nmod":
        return "Cx"
    return {
        "ADJ": "Ca",
        "NOUN": "Cc",
        "PROPN": "Cp",
        "NUM": "Cq",
        "DET": "Cd",
        "PRON": "Ci",
    }.get(pos, "Cx")


def _predicate_type_and_subtype(token: Token) -> str:
    if _is_verb(token):
        return "Pv"
    else:
        return "Px"


def _modifier_type_and_subtype(token: Token) -> str:
    pos: str = token.pos_
    dep: str = token.dep_
    if dep in {"neg", "nk"}:
        return "Mn"
    elif dep in {"poss", "pg", "ag"}:
        return "Mp"
    elif pos == "ADJ":
        return "Ma"
    elif pos == "DET":
        return "Md"
    elif pos == "NUM":
        return "Mq"
    elif pos == "AUX":
        return "Mm"  # modal
    else:
        return "Mx"


def _trigger_type_and_subtype(token: Token) -> str:
    # indirect object
    if token.dep_ in {"iobj", "dative", "obl:arg", "da"}:
        return "Ti"
    else:
        return "Tx"


def _is_verb(token: Token) -> bool:
    return token.pos_ == "VERB"


_BADNESS_WEIGHTS: dict[int, int] = {0: 1000, 1: 100, 2: 10, 3: 1}
_BADNESS_CRASH_PENALTY: int = 1_000_000


@dataclass
class _Beam:
    sequence: list[Hyperedge]
    cum_badness: int
    cum_score: int
    iterations: list[RuleIteration] = field(default_factory=list)
    failed: bool = False


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
            "language": {
                "type": str,
                "default": "en",
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
            "use_atomizer_subtype": {
                "type": bool,
                "default": True,
                "description": (
                    "If True (default), use the atomizer's classification "
                    "directly as the atom type+subtype. If False, refine the "
                    "atomizer's prediction via _concept_type_and_subtype, "
                    "_predicate_type_and_subtype, _modifier_type_and_subtype, "
                    "_trigger_type_and_subtype, and force B/J types to Bx/Jx."
                ),
                "required": False,
            },
            "atomizer_model_path": {
                "type": str,
                "default": None,
                "description": (
                    "Path or Hugging Face repo id to load the atomizer "
                    "model from. Defaults to the bundled "
                    "'hyperquest/atom-classifier' repo."
                ),
                "required": False,
                "is_path": True,
            },
            "beam_width": {
                "type": int,
                "default": 5,
                "description": (
                    "Beam search width for the reduction loop. 1 (default) "
                    "is greedy. Higher values keep multiple parse hypotheses "
                    "alive in parallel and pick the lowest-cumulative-badness "
                    "one at the end (tie-broken by cumulative score)."
                ),
                "required": False,
            },
        }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        self.lang: str = self.params["language"]

        if self.lang not in SPACY_MODELS:
            raise RuntimeError(f"Language code '{self.lang}' is not recognized.")

        debug: bool = self.params.get("debug", False)
        lang_namespace: bool = self.params.get("lang_namespace", False)
        self.atom_lang: str = self.lang if lang_namespace else ""
        self.use_atomizer_subtype: bool = self.params.get("use_atomizer_subtype", True)
        self.atomizer_model_path: str | None = self.params.get("atomizer_model_path")
        self.beam_width: int = max(1, int(self.params.get("beam_width", 5)))

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

        self.alpha: Alpha = Alpha(
            use_atomizer=True,
            use_atomizer_subtype=self.use_atomizer_subtype,
            atomizer_model_path=self.atomizer_model_path,
        )

        self.debug: bool = debug

        self.atom2token: dict[Atom, Token] = {}
        self.orig_atom: dict[Atom, UniqueAtom] = {}
        self.token2atom: dict[Token, Atom] = {}
        self.depths: dict[Atom, int] = {}
        self.connections: set[tuple[Atom, Atom]] = set()
        self.doc: Doc | None = None

        self._repl_session: Any = None
        self._cur_trace: ParseTrace | None = None

    def debug_msg(self, msg: str) -> None:
        if self.debug:
            print(msg)

    def _report_enabled(self) -> bool:
        sess: Any = self._repl_session
        return bool(sess and sess.settings.get("report", False))

    def install_repl(self, session: object) -> None:
        from hyperbase_parser_ab.repl import install

        self._repl_session = session
        install(self, session)

    def parse_sentence(self, sentence: str) -> list[ParseResult]:
        sentence = re.sub(r"\s+", " ", sentence).strip()

        if self.nlp:
            self.orig_atom = {}
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
            self._cur_trace = ParseTrace() if self._report_enabled() else None

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
                if self._cur_trace is not None:
                    self._cur_trace.post_processing.append(("initial", str(edge)))

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
                if self._cur_trace is not None:
                    self._cur_trace.post_processing.append(("arg_roles", str(edge)))
                edge = self._repair(edge)
                self.debug_msg(f"After repair: {edge!s}")
                if self._cur_trace is not None:
                    self._cur_trace.post_processing.append(("repair", str(edge)))
                edge = self._normalise_modifiers(edge)
                self.debug_msg(f"After modifier normalisation: {edge!s}")
                if self._cur_trace is not None:
                    self._cur_trace.post_processing.append(
                        ("normalise_modifiers", str(edge))
                    )
                edge = self._post_process(edge)
                self.debug_msg(f"After post-processing: {edge!s}")
                if self._cur_trace is not None and edge is not None:
                    self._cur_trace.post_processing.append(("post_process", str(edge)))
                if edge is not None:
                    atom2word = self._generate_atom2word(edge, offset=offset)
                    if self._cur_trace is not None:
                        try:
                            raw_final_badness = badness_check(
                                edge, [token.text for token in sent]
                            )
                            self._cur_trace.final_badness = {
                                (k if isinstance(k, str) else str(k)): v
                                for k, v in raw_final_badness.items()
                            }
                        except Exception:
                            self._cur_trace.final_badness = {}

            if edge is None:
                return None

            tok_pos_str = _generate_tok_pos(atom2word, edge)
            extra: dict[str, Any] = {"spacy_sent": sent}
            if self._cur_trace is not None:
                extra["parse_trace"] = self._cur_trace
            return ParseResult(
                edge=edge,
                text=str(sent).strip(),
                tokens=[str(token) for token in sent],
                tok_pos=hedge(tok_pos_str),
                failed=failed,
                extra=extra,
            )
        except Exception as e:
            print(f'Caught exception: {e!s} while parsing: "{sent!s}"')
            traceback.print_exc()
            return None

    def _builder_arg_roles(self, edge: Hyperedge) -> str:
        depth1: int = self._dep_depth(edge[1])
        depth2: int = self._dep_depth(edge[2])
        if depth1 > depth2:
            return "am"
        else:
            return "ma"

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
        elif dep in {
            "iobj",
            "dative",
            "obl",
            "obl:arg",
            "da",
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
            if len(subparts) > 2:
                new_part: str = f"{subparts[0]}.{args_string}.{subparts[2]}"
            else:
                new_part = f"{subparts[0]}.{args_string}"
            new_pred: Atom = pred.replace_atom_part(1, new_part)
            unew_pred: Atom = UniqueAtom(new_pred)
            upred: Atom = UniqueAtom(pred)
            self.atom2token[unew_pred] = self.atom2token[upred]
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
                self.orig_atom[unew_builder] = ubuilder
                new_entity = edge.replace_atom(builder, new_builder, unique=True)

        new_subedges: list[Hyperedge] = [
            self._apply_arg_roles(sub) for sub in new_entity
        ]
        new_entity = hedge(new_subedges)

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

    def _parse_token(self, token: Token, atom_type: str) -> tuple[Atom | None, str]:
        main_type = atom_type[0] if len(atom_type) > 0 else ""

        if main_type == "X":
            return None, atom_type

        if not self.use_atomizer_subtype:
            if main_type == "C":
                atom_type = _concept_type_and_subtype(token)
            elif main_type == "P":
                atom_type = _predicate_type_and_subtype(token)
            elif main_type == "M":
                atom_type = _modifier_type_and_subtype(token)
            elif main_type == "B":
                atom_type = "Bx"
            elif main_type == "T":
                atom_type = _trigger_type_and_subtype(token)
            elif main_type == "J":
                atom_type = "Jx"

        text: str = token.text.lower()
        atom: Atom = build_atom(text, atom_type, self.atom_lang)
        self.debug_msg(f"ATOM: {atom}")

        return atom, atom_type

    def _fix_atom_classifications(
        self, sentence: Span, atom_types: tuple[str, ...] | list[str]
    ) -> list[str]:
        fixed: list[str] = list(atom_types)
        for i, token in enumerate(sentence):
            if self.lang == "en" and token.text == "'s":
                fixed[i] = "Bx"
        return fixed

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
        atom_types: tuple[str, ...] | list[str]
        top_candidates: list[list[tuple[str, float]]]
        atom_types, top_candidates = self.alpha.predict(sentence, features)
        atom_types = self._fix_atom_classifications(sentence, atom_types)

        self.token2atom = {}

        atomseq: list[Atom] = []
        for token, predicted_type, candidates in zip(
            sentence, atom_types, top_candidates, strict=True
        ):
            atom: Atom | None
            refined_type: str
            atom, refined_type = self._parse_token(token, predicted_type)
            if atom:
                uatom: Atom = UniqueAtom(atom)
                self.atom2token[uatom] = token
                self.token2atom[token] = uatom
                self.orig_atom[uatom] = uatom
                atomseq.append(uatom)
            if self._cur_trace is not None:
                self._cur_trace.atoms.append(
                    AtomTrace(
                        token_text=token.text,
                        token_idx=token.i,
                        predicted_type=predicted_type,
                        refined_type=refined_type,
                        final_atom=str(atom) if atom is not None else "",
                        dropped=atom is None,
                        top_candidates=candidates,
                    )
                )
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

        mdepth: int = 9999999
        n: int = 0
        for atom_set in atom_sets:
            for atom in atom_set:
                if atom in self.orig_atom:
                    oatom: Atom = self.orig_atom[atom]
                    if oatom in self.depths:
                        n += 1
                        depth: int = self.depths[oatom]
                        if depth < mdepth:
                            mdepth = depth

        return (10000000 if conn else 0) + (mdepth * 100) + n

    def _candidate_token_atoms(self, edge: Hyperedge) -> frozenset[Atom]:
        return frozenset(a for a in edge.all_atoms() if a in self.orig_atom)

    def _window_connected(self, edges: list[Hyperedge]) -> bool:
        atom_sets: list[list[Atom]] = [e.all_atoms() for e in edges]
        return any(self._are_connected(atom_sets, p) for p in range(len(edges)))

    def _is_relcl_constraint_satisfied(self, new_edge: Hyperedge) -> bool:
        # If the connector P-atom corresponds to a token with dep_='acl:relcl',
        # every token-derived atom in new_edge must come from a dep-descendant
        # of that token (or be the token itself).
        if new_edge.connector_mtype() != "P":
            return True
        pred_atom: Atom | None = new_edge.atom_with_type("P")
        if pred_atom is None:
            return True
        pred_token: Token | None = self.atom2token.get(pred_atom)
        if pred_token is None or pred_token.dep_ != "acl:relcl":
            return True
        allowed_tokens: set[Token] = set(pred_token.subtree)
        for atom in new_edge.all_atoms():
            token: Token | None = self.atom2token.get(atom)
            if token is not None and token not in allowed_tokens:
                return False
        return True

    def _candidate_badness(self, edge: Hyperedge) -> int:
        # 'no-argroles' is skipped: argroles are only assigned by
        # _apply_arg_roles after the reduction loop, so every P/B
        # connector trivially fires it mid-loop.
        total: int = 0
        try:
            for issues in edge.check_correctness().values():
                for err_type, _msg in issues:
                    if err_type == "no-argroles":
                        continue
                    total += _BADNESS_WEIGHTS[0]
            for issues in check_structural_quality(edge).values():
                for _err_type, _msg, severity in issues:
                    total += _BADNESS_WEIGHTS.get(severity, 0)
        except Exception:
            return _BADNESS_CRASH_PENALTY
        return total

    def _expand_beam(self, beam: _Beam) -> list[_Beam]:
        base_iter: RuleIteration = RuleIteration(
            iteration=len(beam.iterations),
            sequence_repr=[str(e) for e in beam.sequence],
        )
        # Parallel lists kept in lockstep until the dominance filter.
        # Per-candidate tuple: (score, new_edge, window_start, pos, badness)
        cand_actions: list[tuple[int, Hyperedge, int, int, int]] = []
        cand_records: list[RuleCandidate] = []
        cand_tokens: list[frozenset[Atom]] = []
        cand_conn: list[bool] = []
        cand_can_dominate: list[bool] = []

        for rule_number, rule in enumerate(RULES):
            window_start: int = rule.size - 1
            for pos in range(window_start, len(beam.sequence)):
                new_edge: Hyperedge | None = apply_rule(rule, beam.sequence, pos)
                if new_edge and self._is_relcl_constraint_satisfied(new_edge):
                    window: list[Hyperedge] = beam.sequence[
                        pos - window_start : pos + 1
                    ]
                    score: int = self._score(window)
                    bad: int = self._candidate_badness(new_edge)
                    cand_records.append(
                        RuleCandidate(
                            rule_index=rule_number,
                            rule_repr=rule_repr(rule, rule_number),
                            pos=pos,
                            score=score,
                            new_edge_repr=str(new_edge),
                            badness=bad,
                        )
                    )
                    cand_actions.append((score, new_edge, window_start, pos, bad))
                    cand_tokens.append(self._candidate_token_atoms(new_edge))
                    cand_conn.append(self._window_connected(window))
                    cand_can_dominate.append(rule.can_dominate)
                else:
                    base_iter.rejections.append((rule_number, pos))

        # Drop A if there exists B != A with A's token-atoms a strict subset
        # of B's AND both with the same connectivity status. Only candidates
        # whose rule has can_dominate=True can act as the dominator B.
        n: int = len(cand_actions)
        keep: list[bool] = [True] * n
        for i in range(n):
            for j in range(n):
                if i == j or not cand_can_dominate[j]:
                    continue
                if cand_conn[i] == cand_conn[j] and cand_tokens[i] < cand_tokens[j]:
                    keep[i] = False
                    break
        cand_actions = [cand_actions[i] for i in range(n) if keep[i]]
        base_iter.candidates = [cand_records[i] for i in range(n) if keep[i]]

        if not cand_actions:
            base_iter.fallback_used = True
            if len(beam.sequence) > 0:
                fallback: Hyperedge = hedge([":/J/.", *beam.sequence[:2]])
                new_sequence: list[Hyperedge] = (
                    [fallback] if fallback else []
                ) + beam.sequence[2:]
            else:
                new_sequence = []
            return [
                _Beam(
                    sequence=new_sequence,
                    cum_badness=beam.cum_badness + _BADNESS_CRASH_PENALTY,
                    cum_score=beam.cum_score,
                    iterations=[*beam.iterations, base_iter],
                    failed=True,
                )
            ]

        # Pick top-k actions from this beam by (badness ASC, score DESC) so
        # one beam can't fan out beyond beam_width children.
        indexed: list[tuple[int, tuple[int, Hyperedge, int, int, int]]] = list(
            enumerate(cand_actions)
        )
        indexed.sort(key=lambda ic: (ic[1][4], -ic[1][0]))
        top = indexed[: self.beam_width]

        extensions: list[_Beam] = []
        for cand_idx, (score, new_edge, window_start, pos, bad) in top:
            iter_for_child: RuleIteration = RuleIteration(
                iteration=base_iter.iteration,
                sequence_repr=base_iter.sequence_repr,
                candidates=[
                    RuleCandidate(
                        rule_index=c.rule_index,
                        rule_repr=c.rule_repr,
                        pos=c.pos,
                        score=c.score,
                        new_edge_repr=c.new_edge_repr,
                        badness=c.badness,
                        is_winner=(i == cand_idx),
                    )
                    for i, c in enumerate(base_iter.candidates)
                ],
                rejections=list(base_iter.rejections),
                fallback_used=False,
            )
            new_sequence = [
                *beam.sequence[: pos - window_start],
                new_edge,
                *beam.sequence[pos + 1 :],
            ]
            extensions.append(
                _Beam(
                    sequence=new_sequence,
                    cum_badness=beam.cum_badness + bad,
                    cum_score=beam.cum_score + score,
                    iterations=[*beam.iterations, iter_for_child],
                    failed=beam.failed,
                )
            )
        return extensions

    def _parse_atom_sequence(
        self, atom_sequence: list[Atom]
    ) -> tuple[list[Hyperedge] | None, bool]:
        beams: list[_Beam] = [
            _Beam(
                sequence=list(atom_sequence),
                cum_badness=0,
                cum_score=0,
            )
        ]

        while not all(len(b.sequence) < 2 for b in beams):
            next_beams: list[_Beam] = []
            for beam in beams:
                if len(beam.sequence) < 2:
                    next_beams.append(beam)
                else:
                    next_beams.extend(self._expand_beam(beam))
            next_beams.sort(key=lambda b: (b.cum_badness, -b.cum_score))
            beams = next_beams[: self.beam_width]

        best: _Beam = beams[0]

        if self._cur_trace is not None:
            self._cur_trace.iterations.extend(best.iterations)

        self.debug_msg(f"Final beam sequence: {best.sequence}")
        self.debug_msg(f"Cumulative badness: {best.cum_badness}")
        self.debug_msg(f"Cumulative score: {best.cum_score}")

        if not best.sequence and best.failed:
            return None, True
        return best.sequence, best.failed

    def get_sentences(self, text: str) -> list[str]:
        if self.nlp:
            doc: Doc = self.nlp(text.strip())
            return [str(sent).strip() for sent in doc.sents]
        else:
            raise RuntimeError("spaCy model failed to initialize.")

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
        _edge = self._process_colon_conjunctions(_edge)
        _edge = self._flatten_conjunctions(_edge)
        return _edge
