import heapq
import math
import multiprocessing as mp
import re
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor
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

# Maximum main probability for the runner-up atomizer label to be considered a
# viable alternative atom assignment in the beam-search seed sequences.
_ALT_LABEL_THRESHOLD: float = 0.99


@dataclass
class _Beam:
    sequence: list[Hyperedge]
    total_badness: int
    total_score: int
    total_distortion: int = 0
    iterations: list[RuleIteration] = field(default_factory=list)
    failed: bool = False
    seq_index: int = 0


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
                    "alive in parallel and pick the lowest-total-badness "
                    "one at the end (tie-broken by total score)."
                ),
                "required": False,
            },
            "exact_search": {
                "type": bool,
                "default": False,
                "description": (
                    "If True, run an exhaustive branch-and-bound search "
                    "instead of beam search. Guarantees finding the "
                    "(badness, distortion)-optimal parse but disregards "
                    "beam_width and is worst-case exponential. In practice "
                    "fast when low-badness parses are easy to find."
                ),
                "required": False,
            },
            "exact_search_timeout": {
                "type": float,
                "default": 10.0,
                "description": (
                    "Per-atomization wall-clock budget (seconds) for exact "
                    "search. When exceeded, returns the best completed "
                    "beam found so far for that atomization, or falls back "
                    "to beam search if none completed. 0 disables the "
                    "timeout. Only used when exact_search=True."
                ),
                "required": False,
            },
            "n_workers": {
                "type": int,
                "default": 1,
                "description": (
                    "Number of worker processes for batch parsing. 1 "
                    "(default) runs everything in the calling process. "
                    "Higher values shard each parse_batch call across that "
                    "many spawned workers, each with its own spaCy + "
                    "atomizer instance. Use to actually utilize multiple "
                    "CPU cores when parsing large corpora."
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
        self.exact_search: bool = bool(self.params.get("exact_search", False))
        self.exact_search_timeout: float = float(
            self.params.get("exact_search_timeout", 30.0)
        )
        self.n_workers: int = max(1, int(self.params.get("n_workers", 1)))
        self._worker_pool: ProcessPoolExecutor | None = None

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
        self.dep_dist: dict[tuple[UniqueAtom, UniqueAtom], int] = {}
        self.doc: Doc | None = None

        self._repl_session: Any = None
        self._cur_trace: ParseTrace | None = None

    def debug_msg(self, msg: str) -> None:
        if self.debug:
            print(msg)

    def _report_enabled(self) -> bool:
        sess: Any = self._repl_session
        return bool(sess and sess.settings.get("report", False))

    def _exact_search_enabled(self) -> bool:
        sess: Any = self._repl_session
        if sess is not None:
            return bool(sess.settings.get("exact_search", self.exact_search))
        return self.exact_search

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

    def parse_batch(self, sentences: list[str]) -> list[list[ParseResult]]:
        """Parse multiple sentences with shared spaCy + atomizer passes.

        spaCy runs as a single ``nlp.pipe`` over the whole batch, the
        atomizer runs as a single padded forward pass, and the
        per-sentence beam search is then optionally sharded across
        ``n_workers`` processes. Falls back to the base-class
        sentence-by-sentence loop if input is empty."""
        if not sentences:
            return []
        if self.n_workers > 1 and len(sentences) > 1:
            return self._parse_batch_parallel(sentences)
        return self._parse_batch_inproc(sentences)

    def _parse_batch_inproc(self, sentences: list[str]) -> list[list[ParseResult]]:
        if self.nlp is None:
            raise RuntimeError("spaCy model failed to initialize.")

        normalized: list[str] = [re.sub(r"\s+", " ", s).strip() for s in sentences]

        # Single nlp.pipe call replaces N self.nlp() invocations.
        docs: list[Doc] = list(self.nlp.pipe(normalized))

        # Flatten input -> sub-sentences, preserving offsets and the
        # parent input index so we can regroup results at the end.
        flat: list[tuple[int, Span, int]] = []
        for input_idx, doc in enumerate(docs):
            offset: int = 0
            for sent in doc.sents:
                flat.append((input_idx, sent, offset))
                offset += len(sent)

        results_per_input: list[list[ParseResult]] = [[] for _ in sentences]
        if not flat:
            return results_per_input

        spans: list[Span] = [s for _, s, _ in flat]
        features_list: list[list[tuple[str, str, str, str, str]]] = [
            self._token_features(s) for s in spans
        ]

        assert self.alpha is not None, "Alpha must be initialized before parsing"
        # One batched atomizer forward pass for every sub-sentence.
        predictions = self.alpha.predict_batch(spans, features_list)

        cur_input_idx: int = -1
        for (input_idx, sent, sent_offset), pred in zip(flat, predictions, strict=True):
            # Match parse_sentence semantics: orig_atom is reset once
            # per input string, then accumulates across that string's
            # sub-sentences.
            if input_idx != cur_input_idx:
                self.orig_atom = {}
                self.doc = docs[input_idx]
                cur_input_idx = input_idx
            try:
                parse = self.parse_spacy_sentence(
                    sent, offset=sent_offset, prediction=pred
                )
            except RuntimeError as error:
                print(error)
                parse = None
            if parse:
                results_per_input[input_idx].append(parse)
        return results_per_input

    def _ensure_worker_pool(self) -> ProcessPoolExecutor:
        if self._worker_pool is None:
            ctx = mp.get_context("spawn")
            self._worker_pool = ProcessPoolExecutor(
                max_workers=self.n_workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(self.params,),
            )
        return self._worker_pool

    def _parse_batch_parallel(self, sentences: list[str]) -> list[list[ParseResult]]:
        pool = self._ensure_worker_pool()
        n: int = len(sentences)
        n_workers: int = min(self.n_workers, n)
        # Even contiguous shards: every worker gets ceil(n/k) or
        # floor(n/k) sentences. Each shard is large enough for the
        # in-worker batched spaCy + atomizer passes to amortize.
        chunk_size: int = (n + n_workers - 1) // n_workers
        chunks: list[list[str]] = [
            sentences[i : i + chunk_size] for i in range(0, n, chunk_size)
        ]
        chunk_results: list[list[list[ParseResult]]] = list(
            pool.map(_worker_parse_chunk, chunks)
        )
        # Flatten back in submission order.
        out: list[list[ParseResult]] = []
        for cr in chunk_results:
            out.extend(cr)
        return out

    def close(self) -> None:
        """Shut down the worker pool, if one was started. Idempotent.

        Not called from ``__del__``; ``ProcessPoolExecutor`` registers
        its own atexit hook that handles interpreter shutdown, and
        racing it from a finalizer leads to spurious crashes."""
        if self._worker_pool is not None:
            self._worker_pool.shutdown(wait=True)
            self._worker_pool = None

    def parse_spacy_sentence(
        self,
        sent: Span,
        atom_sequence: list[Atom] | None = None,
        offset: int = 0,
        prediction: tuple[tuple[str, ...] | list[str], list[list[tuple[str, float]]]]
        | None = None,
    ) -> ParseResult | None:
        try:
            self._cur_trace = ParseTrace() if self._report_enabled() else None

            atom_sequences: list[list[Atom]]
            sequence_traces: list[list[AtomTrace]]
            if atom_sequence is None:
                atom_sequences, sequence_traces = self._build_atom_sequences(
                    sent, prediction=prediction
                )
            else:
                atom_sequences = [atom_sequence]
                sequence_traces = [[]]

            self._compute_depths_and_connections(sent.root)
            self._compute_dep_distances()

            edge: Hyperedge | None = None
            result: list[Hyperedge] | None
            failed: bool
            winner_idx: int
            result, failed, winner_idx = self._parse_atom_sequence(atom_sequences)

            if self._cur_trace is not None and sequence_traces:
                idx: int = winner_idx if 0 <= winner_idx < len(sequence_traces) else 0
                self._cur_trace.atoms.extend(sequence_traces[idx])
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
            # "ROOT",
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
            "obl:mod",
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

    def _kbest_label_states(
        self,
        per_token_choices: list[list[tuple[str, float]]],
        k: int,
    ) -> list[tuple[int, ...]]:
        """Enumerate up to k joint label assignments in order of descending
        joint probability, using a best-first priority queue over the
        product of independently-ranked per-token candidate lists.

        Each state is a tuple ``(j_0, ..., j_{n-1})`` where ``j_i`` selects
        which candidate is used at token position i. Priority is the
        negative log joint probability (lower = better)."""
        n: int = len(per_token_choices)
        if n == 0:
            return [()]

        def cost(state: tuple[int, ...]) -> float:
            c: float = 0.0
            for i, j in enumerate(state):
                score: float = per_token_choices[i][j][1]
                if score <= 0.0:
                    return float("inf")
                c += -math.log(score)
            return c

        start: tuple[int, ...] = (0,) * n
        heap: list[tuple[float, int, tuple[int, ...]]] = []
        # The integer counter breaks ties so heapq never compares the
        # tuples themselves (which would be wasted work).
        counter: int = 0
        heapq.heappush(heap, (cost(start), counter, start))
        visited: set[tuple[int, ...]] = {start}
        results: list[tuple[int, ...]] = []

        while heap and len(results) < k:
            _, _, state = heapq.heappop(heap)
            results.append(state)
            for i in range(n):
                if state[i] + 1 < len(per_token_choices[i]):
                    new_state: tuple[int, ...] = (
                        *state[:i],
                        state[i] + 1,
                        *state[i + 1 :],
                    )
                    if new_state not in visited:
                        visited.add(new_state)
                        counter += 1
                        heapq.heappush(heap, (cost(new_state), counter, new_state))

        return results

    @staticmethod
    def _token_features(
        sentence: Span,
    ) -> list[tuple[str, str, str, str, str]]:
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
        return features

    def _build_atom_sequences(
        self,
        sentence: Span,
        prediction: tuple[tuple[str, ...] | list[str], list[list[tuple[str, float]]]]
        | None = None,
    ) -> tuple[list[list[Atom]], list[list[AtomTrace]]]:
        atom_types: tuple[str, ...] | list[str]
        top_candidates: list[list[tuple[str, float]]]
        if prediction is not None:
            atom_types, top_candidates = prediction
        else:
            features: list[tuple[str, str, str, str, str]] = self._token_features(
                sentence
            )
            assert self.alpha is not None, "Alpha must be initialized before parsing"
            atom_types, top_candidates = self.alpha.predict(sentence, features)
        atom_types = self._fix_atom_classifications(sentence, atom_types)

        self.token2atom = {}

        # Per-token choice lists. Element 0 is always the (fixed) top-1
        # label; element 1 (when present) is the runner-up label if top-1
        # score < _ALT_LABEL_THRESHOLD, differs from top-1, and yields a
        # non-None atom. Excluding alternatives that drop the token keeps
        # the sequence length identical across variants.
        per_token_choices: list[list[tuple[str, float]]] = []
        for token, top1_label, candidates in zip(
            sentence, atom_types, top_candidates, strict=True
        ):
            top1_score: float = candidates[0][1] if candidates else 1.0
            choices: list[tuple[str, float]] = [(top1_label, top1_score)]
            if len(candidates) >= 2:
                top2_label, top2_score = candidates[1]
                if top1_score < _ALT_LABEL_THRESHOLD and top2_label != top1_label:
                    alt_atom, _alt_refined = self._parse_token(token, top2_label)
                    if alt_atom is not None:
                        choices.append((top2_label, top2_score))
            per_token_choices.append(choices)

        state_vectors: list[tuple[int, ...]] = self._kbest_label_states(
            per_token_choices, self.beam_width
        )
        # Ensure the all-top-1 state comes first so its atoms become the
        # canonical mapping in self.token2atom (used downstream by
        # _compute_depths_and_connections).
        zero_state: tuple[int, ...] = (0,) * len(per_token_choices)
        if state_vectors and state_vectors[0] != zero_state:
            if zero_state in state_vectors:
                state_vectors.remove(zero_state)
            state_vectors.insert(0, zero_state)

        primary_atoms_per_token: list[Atom | None] = [None] * len(per_token_choices)
        sequences: list[list[Atom]] = []
        sequence_traces: list[list[AtomTrace]] = []

        for seq_idx, state in enumerate(state_vectors):
            atomseq: list[Atom] = []
            atom_traces: list[AtomTrace] = []
            for tok_idx, (token, choices, j) in enumerate(
                zip(sentence, per_token_choices, state, strict=True)
            ):
                label, _score = choices[j]
                atom: Atom | None
                refined_type: str
                atom, refined_type = self._parse_token(token, label)
                if atom is not None:
                    uatom: Atom = UniqueAtom(atom)
                    self.atom2token[uatom] = token
                    if seq_idx == 0:
                        self.token2atom[token] = uatom
                        self.orig_atom[uatom] = uatom
                        primary_atoms_per_token[tok_idx] = uatom
                    else:
                        # Map alternative atoms back to the primary so
                        # _is_pair_connected / _dep_depth see the same
                        # token-level connectivity as the primary atom.
                        primary = primary_atoms_per_token[tok_idx]
                        primary = (
                            cast(UniqueAtom, primary) if primary is not None else None
                        )
                        self.orig_atom[uatom] = (
                            primary if primary is not None else uatom
                        )
                    atomseq.append(uatom)
                atom_traces.append(
                    AtomTrace(
                        token_text=token.text,
                        token_idx=token.i,
                        predicted_type=label,
                        refined_type=refined_type,
                        final_atom=str(atom) if atom is not None else "",
                        dropped=atom is None,
                        top_candidates=top_candidates[tok_idx],
                        chosen_label_rank=j,
                    )
                )
            sequences.append(atomseq)
            sequence_traces.append(atom_traces)

        self.debug_msg(f"Atom sequences ({len(sequences)}): {sequences}")
        return sequences, sequence_traces

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

    def _compute_dep_distances(self) -> None:
        # BFS over self.connections (bidirectional) to populate the
        # all-pairs distance matrix between UniqueAtom nodes.
        self.dep_dist = {}
        adj: dict[UniqueAtom, list[UniqueAtom]] = {}
        for a, b in self.connections:
            ua_a = self.orig_atom.get(a)
            ua_b = self.orig_atom.get(b)
            if ua_a is None or ua_b is None:
                continue
            adj.setdefault(ua_a, []).append(ua_b)

        for src in adj:
            dist: dict[UniqueAtom, int] = {src: 0}
            queue: deque[UniqueAtom] = deque([src])
            while queue:
                u = queue.popleft()
                for v in adj.get(u, ()):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for ua, d in dist.items():
                self.dep_dist[(src, ua)] = d

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

    def _atom_depths(self, edge: Hyperedge) -> dict[UniqueAtom, int]:
        # Tree-distance from the root of edge's hyperedge-tree to every
        # atom in edge that maps to a UniqueAtom. The tree of (C e1 ... ek)
        # has root = root_atom(C); atoms in T(C) keep their depth, atoms in
        # T(ei) for i>=1 have an extra +1 hop (root(T(ei)) -> root(T(C))).
        out: dict[UniqueAtom, int] = {}
        self._collect_atom_depths(edge, 0, out)
        return out

    def _collect_atom_depths(
        self, edge: Hyperedge, base_depth: int, out: dict[UniqueAtom, int]
    ) -> None:
        if edge.atom:
            ua = self.orig_atom.get(cast(Atom, edge))
            if ua is not None and (ua not in out or base_depth < out[ua]):
                out[ua] = base_depth
            return
        for i, child in enumerate(edge):
            child_base = base_depth + (0 if i == 0 else 1)
            self._collect_atom_depths(child, child_base, out)

    def _distance_distortion_delta(self, new_edge: Hyperedge) -> int:
        # Sum |dep_dist(a, b) - hyper_dist(a, b)| over every pair (a, b)
        # whose endpoints first co-occur in this hyperedge. A pair first
        # co-occurs when its atoms sit in distinct immediate children of
        # new_edge. Once locked in, hyper_dist(a, b) is invariant under
        # subsequent nesting, so charging each pair once at its first
        # co-occurrence yields a well-defined cumulative cost.
        #
        # Each pair's contribution is scaled by the size of new_edge
        # (its total atom count), so mistakes inside a large hyperedge
        # cost more than the same mistake inside a tiny one.
        if new_edge.atom:
            return 0
        children: list[Hyperedge] = list(new_edge)
        if len(children) < 2:
            return 0
        sub_depths: list[dict[UniqueAtom, int]] = [
            self._atom_depths(c) for c in children
        ]
        edge_size: int = len(new_edge.all_atoms())

        distortion: int = 0
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                # Cross-child hyper distance: depth_in_child_i + depth_in_child_j
                # plus 1 hop if one side is the connector subtree (i==0),
                # else 2 hops (both arg subtrees route through the connector).
                extra: int = 1 if (i == 0 or j == 0) else 2
                for ua_a, da in sub_depths[i].items():
                    for ua_b, db in sub_depths[j].items():
                        if ua_a == ua_b:
                            continue
                        dep_d = self.dep_dist.get((ua_a, ua_b))
                        if dep_d is None:
                            continue
                        hyper_d = da + db + extra
                        distortion += abs(dep_d - hyper_d) * edge_size
        return distortion

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

    def _expand_beam(
        self,
        beam: _Beam,
        prune: bool = True,
        best_so_far: tuple[float, float] = (math.inf, math.inf),
    ) -> list[_Beam]:
        base_iter: RuleIteration = RuleIteration(
            iteration=len(beam.iterations),
            sequence_repr=[str(e) for e in beam.sequence],
        )
        # Parallel lists kept in lockstep until the dominance filter.
        # Per-candidate tuple:
        # (score, new_edge, window_start, pos, badness, distortion)
        cand_actions: list[tuple[int, Hyperedge, int, int, int, int]] = []
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
                    distortion: int = self._distance_distortion_delta(new_edge)
                    cand_records.append(
                        RuleCandidate(
                            rule_index=rule_number,
                            rule_repr=rule_repr(rule, rule_number),
                            pos=pos,
                            score=score,
                            new_edge_repr=str(new_edge),
                            badness=bad,
                            distortion=distortion,
                        )
                    )
                    cand_actions.append(
                        (score, new_edge, window_start, pos, bad, distortion)
                    )
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
                    total_badness=beam.total_badness + _BADNESS_CRASH_PENALTY,
                    total_score=beam.total_score,
                    total_distortion=beam.total_distortion,
                    iterations=[*beam.iterations, base_iter],
                    failed=True,
                    seq_index=beam.seq_index,
                )
            ]

        indexed: list[tuple[int, tuple[int, Hyperedge, int, int, int, int]]] = list(
            enumerate(cand_actions)
        )
        if prune:
            # Pick top-k actions from this beam by (badness ASC, score DESC)
            # so one beam can't fan out beyond beam_width children.
            indexed.sort(key=lambda ic: (ic[1][4], -ic[1][0]))
            top = indexed[: self.beam_width]
        else:
            # Branch-and-bound: keep every candidate whose extension cost
            # is not already strictly worse than best_so_far. Cumulative
            # badness and distortion are monotone non-decreasing, so this
            # prune is admissible.
            top = [
                ia
                for ia in indexed
                if (
                    beam.total_badness + ia[1][4],
                    beam.total_distortion + ia[1][5],
                )
                <= best_so_far
            ]

        extensions: list[_Beam] = []
        for cand_idx, (
            score,
            new_edge,
            window_start,
            pos,
            bad,
            distortion,
        ) in top:
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
                        distortion=c.distortion,
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
                    total_badness=beam.total_badness + bad,
                    total_score=beam.total_score + score,
                    total_distortion=beam.total_distortion + distortion,
                    iterations=[*beam.iterations, iter_for_child],
                    failed=beam.failed,
                    seq_index=beam.seq_index,
                )
            )
        return extensions

    def _exact_search(
        self, beams: list[_Beam], deadline: float | None = None
    ) -> tuple[list[_Beam], bool]:
        # Branch-and-bound search guaranteeing the (badness, distortion)-
        # optimal parse. cumulative badness and distortion are both
        # monotone non-decreasing across reductions, so once any beam
        # completes we can prune every in-flight beam whose current cost
        # already exceeds the best completed cost.
        #
        # If `deadline` is provided (time.monotonic() target), the search
        # bails out when reached, returning whatever completed beams it
        # has found and `finished=False`.
        best_so_far: tuple[float, float] = (math.inf, math.inf)

        while not all(len(b.sequence) < 2 for b in beams):
            if deadline is not None and time.monotonic() >= deadline:
                completed = [b for b in beams if len(b.sequence) < 2]
                return completed, False
            next_beams: list[_Beam] = []
            for beam in beams:
                if len(beam.sequence) < 2:
                    next_beams.append(beam)
                    continue
                if (beam.total_badness, beam.total_distortion) > best_so_far:
                    continue
                for ext in self._expand_beam(
                    beam, prune=False, best_so_far=best_so_far
                ):
                    cost = (ext.total_badness, ext.total_distortion)
                    if cost > best_so_far:
                        continue
                    if len(ext.sequence) < 2 and cost < best_so_far:
                        best_so_far = cost
                    next_beams.append(ext)
            # best_so_far may have improved during this iteration; re-prune.
            beams = [
                b
                for b in next_beams
                if (b.total_badness, b.total_distortion) <= best_so_far
            ]
            if not beams:
                break

        return beams, True

    def _beam_search_one_sequence(
        self, atom_sequence: list[Atom], seq_index: int
    ) -> list[_Beam]:
        beams: list[_Beam] = [
            _Beam(
                sequence=list(atom_sequence),
                total_badness=0,
                total_score=0,
                seq_index=seq_index,
            )
        ]
        while not all(len(b.sequence) < 2 for b in beams):
            next_beams: list[_Beam] = []
            for beam in beams:
                if len(beam.sequence) < 2:
                    next_beams.append(beam)
                else:
                    next_beams.extend(self._expand_beam(beam))
            next_beams.sort(key=lambda b: (b.total_badness, -b.total_score))
            beams = next_beams[: self.beam_width]
        return beams

    def _parse_atom_sequence(
        self, atom_sequences: list[list[Atom]]
    ) -> tuple[list[Hyperedge] | None, bool, int]:
        beams: list[_Beam] = []
        if self._exact_search_enabled():
            # Per-sequence exact search with a per-sequence wall-clock
            # budget. If the budget expires before this atomization's
            # search completes, we fall back to beam search for it (or
            # use whatever exact completions were found, if any). This
            # prevents one runaway atomization from blocking the batch.
            timeout = self.exact_search_timeout
            for i, seq in enumerate(atom_sequences):
                seed = [
                    _Beam(
                        sequence=list(seq),
                        total_badness=0,
                        total_score=0,
                        seq_index=i,
                    )
                ]
                deadline = time.monotonic() + timeout if timeout > 0 else None
                seq_beams, finished = self._exact_search(seed, deadline=deadline)
                if not finished:
                    if seq_beams:
                        print(
                            f"[parser] exact_search timed out for seq_idx={i}"
                            f" after {timeout}s; using best completed beam",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"[parser] exact_search timed out for seq_idx={i}"
                            f" after {timeout}s; falling back to beam search",
                            file=sys.stderr,
                        )
                        seq_beams = self._beam_search_one_sequence(seq, i)
                beams.extend(seq_beams)
        else:
            # Beam search: run each atomization in its own search so
            # alternatives don't compete for the same beam_width slots.
            # Otherwise a sequence whose optimum requires intermediate
            # states that score worse than a competitor at iteration t
            # gets pruned before its better final state is realized.
            for i, seq in enumerate(atom_sequences):
                beams.extend(self._beam_search_one_sequence(seq, i))

        beams.sort(key=lambda b: (b.total_badness, b.total_distortion))
        best: _Beam = beams[0]

        if self._cur_trace is not None:
            self._cur_trace.iterations.extend(best.iterations)

        self.debug_msg(f"Final beam sequence: {best.sequence}")
        self.debug_msg(f"Total badness: {best.total_badness}")
        self.debug_msg(f"Total score: {best.total_score}")
        self.debug_msg(f"Total distortion: {best.total_distortion}")
        self.debug_msg(f"Winning seq_index: {best.seq_index}")
        if len(beams) > 1:
            self.debug_msg(f"Surviving beams ({len(beams)}):")
            for b in beams:
                self.debug_msg(
                    f"  seq_index={b.seq_index} "
                    f"badness={b.total_badness} "
                    f"distortion={b.total_distortion} "
                    f"score={b.total_score}"
                )

        if not best.sequence and best.failed:
            return None, True, best.seq_index
        return best.sequence, best.failed, best.seq_index

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
            if "s" not in ars:
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
                if ar == "?":
                    if subedge.mt in {"R", "S"}:
                        _ar = "x"
                    elif subedge.mt == "C":
                        if "s" not in ars:
                            _ar = "s"
                        elif "o" not in ars:
                            _ar = "o"
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


# Worker-process state. Each spawned process gets its own AlphaBetaParser
# instance, lazily constructed by _worker_init on pool startup.
_WORKER_PARSER: AlphaBetaParser | None = None


def _worker_init(params: dict[str, Any]) -> None:
    import os

    global _WORKER_PARSER
    # HF tokenizers warns loudly when forked after first use; we don't
    # need its internal thread pool inside the workers.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Force the worker to single-process; otherwise it would try to
    # spawn its own pool.
    worker_params: dict[str, Any] = dict(params)
    worker_params["n_workers"] = 1
    _WORKER_PARSER = AlphaBetaParser(worker_params)


def _worker_parse_chunk(chunk: list[str]) -> list[list[ParseResult]]:
    assert _WORKER_PARSER is not None, "worker parser not initialized"
    results: list[list[ParseResult]] = _WORKER_PARSER._parse_batch_inproc(chunk)
    # spaCy Span objects don't survive cross-process pickling cleanly;
    # the REPL hook that consumes 'spacy_sent' only runs in the main
    # process anyway, so drop it from worker results.
    for sent_results in results:
        for r in sent_results:
            r.extra.pop("spacy_sent", None)
    return results
