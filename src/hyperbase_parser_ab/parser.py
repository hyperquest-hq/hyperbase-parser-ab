import itertools
import multiprocessing as mp
import re
import traceback
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
from hyperbase_parser_ab.rules import RULES, Rule, apply_rule_indices
from hyperbase_parser_ab.trace import (
    AtomTrace,
    ParseTrace,
    RuleCandidate,
    RuleIteration,
    SubstitutionRound,
    SubstitutionTrial,
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
class _ParseState:
    sequence: list[Hyperedge]
    total_badness: int
    total_score: int
    total_distortion: int = 0
    iterations: list[RuleIteration] = field(default_factory=list)
    failed: bool = False
    seq_index: int = 0


@dataclass
class _AtomAlt:
    # A pre-built runner-up atom for an uncertain token. Both the primary
    # and the alt are already registered in self.atom2token /
    # self.orig_atom by _build_seed_atomization, so the hill-climb code
    # only has to swap seed_seq[seq_pos] = uatom for a trial.
    tok_idx: int
    seq_pos: int
    uatom: "UniqueAtom"
    label: str
    refined_type: str
    final_atom_str: str
    top1_prob: float


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
                "reload": False,
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
            "uncertain_atom_ratio": {
                "type": float,
                "default": 0,
                "description": (
                    "Fraction of a sentence's *eligible* tokens treated as "
                    "uncertain. A token is eligible only when its top-2 "
                    "atomizer label has a different main type (first "
                    "character) from top-1 — same-main-type runner-ups "
                    "(e.g. Cd/Cx) are considered fixed at top-1. Eligible "
                    "tokens are ranked ascending by top-1 probability and "
                    "the bottom `int(ratio * len(eligible))` get the "
                    "runner-up label as a viable alternative in seed "
                    "sequences. 0 disables alternatives; 1 considers "
                    "every eligible token uncertain."
                ),
                "required": False,
                "reload": False,
            },
            "post_processing": {
                "type": bool,
                "default": True,
                "description": (
                    "If True (default), apply the post-search transforms "
                    "(_apply_arg_roles, _deepen_modifiers) to the reduced "
                    "edge before returning. If False, return the raw edge "
                    "unchanged. Useful for debugging the search output."
                ),
                "required": False,
                "reload": False,
            },
        }

    def apply_live_setting(self, name: str, value: Any) -> None:  # noqa: ANN401
        if name == "uncertain_atom_ratio":
            value = min(1.0, max(0.0, float(value)))
        super().apply_live_setting(name, value)

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
        self.n_workers: int = max(1, int(self.params.get("n_workers", 1)))
        self.uncertain_atom_ratio: float = min(
            1.0, max(0.0, float(self.params.get("uncertain_atom_ratio", 0)))
        )
        self.post_processing: bool = bool(self.params.get("post_processing", True))
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
        self._dpt_pairs: set[frozenset[UniqueAtom]] = set()
        self._dpt_directed: set[tuple[UniqueAtom, UniqueAtom]] = set()
        self._dpt_parent_of: dict[UniqueAtom, UniqueAtom] = {}
        self._dpt_children_of: dict[UniqueAtom, set[UniqueAtom]] = {}
        # Criterion 3.12 — X/J separator splits per DPT parent. Each
        # entry maps a parent atom to a list of (left_set, right_set)
        # pairs, one per X- or J-labeled separator in the parent's
        # spaCy children sequence. `left_set` always contains the parent
        # atom (parent belongs to the first separated group).
        self._dpt_separators: dict[
            UniqueAtom, list[tuple[frozenset[UniqueAtom], frozenset[UniqueAtom]]]
        ] = {}
        # Atoms found to be "stranded" — i.e. left bare at the top of the
        # final hyperedge or force-joined via the :/J/. iteration-level
        # fallback — at a previous pass of the current parse. Drives the
        # sliding-window per-position fallback inside _expand_state: that
        # block fires only for positions whose entity carries one of these
        # atoms. Empty on the first pass; grown across passes by
        # _detect_stranded until it stabilises.
        self._stranded_atoms: set[UniqueAtom] = set()
        # Atoms consumed by the :/J/. iteration-level fallback during the
        # current pass. Captured at the fallback site in _expand_state,
        # reset between passes.
        self._force_joined_atoms: set[UniqueAtom] = set()
        # Most recently parsed sub-sentence text, captured for the
        # /dpt REPL diagnostic command (which re-parses and walks the
        # result for per-edge distortion analysis). Populated on every
        # parse_spacy_sentence call.
        self._last_parsed_sentence: str | None = None
        # Populated at the end of parse_spacy_sentence (only on
        # non-forced runs) so the /sub <number> REPL command can replay
        # any trial's parse.
        # Shape: {"sentence": str,
        #         "trial_subs": dict[int, dict[int, str]]}
        # where trial_subs maps each global trial number to its
        # cumulative {tok_idx: alt_label} substitutions.
        self._last_sub_climb: dict[str, Any] | None = None
        self.doc: Doc | None = None

        self._repl_session: Any = None
        self._cur_trace: ParseTrace | None = None
        self._cur_sequence: list[Hyperedge] | None = None

    def debug_msg(self, msg: str) -> None:
        if self.debug:
            print(msg)

    def _report_enabled(self) -> bool:
        sess: Any = self._repl_session
        return bool(sess and sess.settings.get("report", False))

    def _post_processing_enabled(self) -> bool:
        sess: Any = self._repl_session
        if sess is not None:
            return bool(sess.settings.get("post_processing", self.post_processing))
        return self.post_processing

    def install_repl(self, session: object) -> None:
        from hyperbase_parser_ab.repl import install

        self._repl_session = session
        install(self, session)

    def _detect_stranded(self, final_edge: Hyperedge) -> set[UniqueAtom]:
        # Returns the set of original atoms that "never satisfied a rule"
        # in the just-completed pass: atoms at depth 1 of `final_edge`
        # whose mtype is pivot-capable (so the failure is a real one, not
        # a legitimate punctuation leaf) plus atoms that the
        # iteration-level :/J/. fallback in _expand_state force-joined
        # during the pass (captured into self._force_joined_atoms).
        pivot_capable: set[str] = {r.first_type for r in RULES}
        stranded: set[UniqueAtom] = set()
        if not final_edge.atom:
            for child in final_edge:
                if child.atom and child.mtype() in pivot_capable:
                    ua = self.orig_atom.get(cast(Atom, child))
                    if ua is not None:
                        stranded.add(ua)
        stranded |= self._force_joined_atoms
        return stranded

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

    def parse_sentence_with_substitutions(
        self, sentence: str, forced_subs: dict[int, str]
    ) -> list[ParseResult]:
        """Parse `sentence` but force the atomizer label at each `tok_idx`
        to the corresponding `alt_label` in `forced_subs`. Hill-climbing
        is skipped — exactly one reduction runs per sub-sentence with
        the patched seed. Used by the REPL `/sub <N>` command to replay
        a recorded substitution trial. The replay does not update
        `self._last_sub_climb`, so subsequent `/sub` calls still refer
        to the original (non-forced) hill-climb."""
        sentence = re.sub(r"\s+", " ", sentence).strip()

        if not self.nlp:
            raise RuntimeError("spaCy model failed to initialize.")
        self.orig_atom = {}
        parses: list[ParseResult] = []
        try:
            self.doc = self.nlp(sentence)
            offset: int = 0
            for sent in self.doc.sents:
                parse: ParseResult | None = self.parse_spacy_sentence(
                    sent, offset=offset, forced_substitutions=forced_subs
                )
                if parse:
                    parses.append(parse)
                offset += len(sent)
        except RuntimeError as error:
            print(error)
        return parses

    def parse_batch(self, sentences: list[str]) -> list[list[ParseResult]]:
        """Parse multiple sentences with shared spaCy + atomizer passes.

        spaCy runs as a single ``nlp.pipe`` over the whole batch, the
        atomizer runs as a single padded forward pass, and the
        per-sentence reduction is then optionally sharded across
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
        forced_substitutions: dict[int, str] | None = None,
    ) -> ParseResult | None:
        try:
            self._cur_trace = ParseTrace() if self._report_enabled() else None
            self._last_parsed_sentence = str(sent)

            seed_seq: list[Atom]
            atom_traces: list[AtomTrace]
            alts: list[_AtomAlt]
            if atom_sequence is None:
                seed_seq, atom_traces, alts = self._build_seed_atomization(
                    sent, prediction=prediction
                )
            else:
                seed_seq = list(atom_sequence)
                atom_traces = []
                alts = []

            self._compute_depths_and_connections(sent.root)
            self._compute_dpt_pairs()
            self._compute_dpt_separators(sent.root)

            # Replay path (used by the REPL `/sub <number>` command):
            # patch the seed at the requested positions and skip
            # hill-climbing so the user sees the parse of *that exact*
            # substitution set.
            if forced_substitutions:
                self._apply_forced_substitutions(
                    sent, seed_seq, alts, atom_traces, forced_substitutions
                )

            edge: Hyperedge | None = None
            best_state: _ParseState | None = None
            substituted: set[int] = set()
            sub_rounds: list[SubstitutionRound] = []
            # Two-pass orchestration. First attempt runs vanilla (empty
            # _stranded_atoms keeps the sliding-window block in
            # _expand_state a no-op). After each attempt we look at the
            # raw final edge plus any atoms force-joined by the :/J/.
            # fallback and add them to _stranded_atoms. If the stranded
            # set grew, re-run the search so the sliding-window block
            # can rescue those atoms. Stop once the set stabilises or a
            # cap is hit.
            self._stranded_atoms = set()
            max_passes: int = 4
            for _attempt in range(max_passes):
                self._force_joined_atoms = set()
                if forced_substitutions is not None:
                    best_state = self._reduce_sequence(seed_seq)
                    substituted = set(forced_substitutions.keys())
                    sub_rounds = []
                else:
                    best_state, substituted, sub_rounds = self._hill_climb_atomization(
                        seed_seq, alts, atom_traces=atom_traces
                    )
                if (
                    best_state is None
                    or best_state.failed
                    or len(best_state.sequence) != 1
                ):
                    raw_final = None
                else:
                    raw_final = non_unique(best_state.sequence[0])
                if raw_final is None:
                    new_stranded = set(self._force_joined_atoms)
                else:
                    new_stranded = self._detect_stranded(raw_final)
                if new_stranded <= self._stranded_atoms:
                    break
                self._stranded_atoms |= new_stranded

            # Reflect any locked-in substitutions in the per-token traces
            # before they're attached: chosen_label_rank=1 marks the alt
            # was used, and we mirror its label / refined type / atom
            # string so the REPL atoms panel matches the real winner.
            # In the forced-substitution replay path,
            # _apply_forced_substitutions has already patched
            # atom_traces, so skip this block to avoid clobbering the
            # forced label with whatever happens to be in `alts`.
            if forced_substitutions is None and atom_traces and substituted:
                alt_by_idx: dict[int, _AtomAlt] = {a.tok_idx: a for a in alts}
                for tok_idx in substituted:
                    a = alt_by_idx.get(tok_idx)
                    if a is None:
                        continue
                    atom_traces[tok_idx].chosen_label_rank = 1
                    atom_traces[tok_idx].predicted_type = a.label
                    atom_traces[tok_idx].refined_type = a.refined_type
                    atom_traces[tok_idx].final_atom = a.final_atom_str

            if self._cur_trace is not None and atom_traces:
                self._cur_trace.atoms.extend(atom_traces)

            result: list[Hyperedge] | None = (
                best_state.sequence if best_state is not None else None
            )
            failed: bool = best_state.failed if best_state is not None else True
            if best_state is not None and self._cur_trace is not None:
                self._cur_trace.iterations.extend(best_state.iterations)
                self._cur_trace.total_badness = best_state.total_badness
                self._cur_trace.total_distortion = best_state.total_distortion
                self._cur_trace.substitution_rounds.extend(sub_rounds)

            # Cache the trial-number -> substitutions map so the REPL
            # `/sub <N>` command can replay any trial. Only update on
            # non-forced runs — otherwise /sub would invalidate itself
            # every time it fires.
            if forced_substitutions is None and sub_rounds:
                trial_subs: dict[int, dict[int, str]] = {}
                for rnd in sub_rounds:
                    for trial in rnd.trials:
                        trial_subs[trial.number] = dict(trial.substitutions)
                self._last_sub_climb = {
                    "sentence": str(sent),
                    "trial_subs": trial_subs,
                }

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
                if self._post_processing_enabled():
                    edge = self._deepen_modifiers(edge)
                    self.debug_msg(f"After deepening modifiers: {edge!s}")
                    if self._cur_trace is not None:
                        self._cur_trace.post_processing.append(
                            ("deepen_modifiers", str(edge))
                        )
                else:
                    self.debug_msg("Post-processing disabled — keeping raw edge.")
                    if self._cur_trace is not None:
                        self._cur_trace.post_processing.append(
                            ("post_processing_disabled", str(edge))
                        )
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

    def _has_flat_name(self, edge: Hyperedge) -> bool:
        for atom in edge.all_atoms():
            uatom: Hyperedge | None = unique(atom)
            if uatom is None:
                continue
            uatom = cast(Atom, uatom)
            token: Token | None = self.atom2token.get(uatom)
            if token is not None and token.dep_ == "flat:name":
                return True
        return False

    def _builder_arg_roles(self, edge: Hyperedge) -> str:
        flat1: bool = self._has_flat_name(edge[1])
        flat2: bool = self._has_flat_name(edge[2])
        if flat1 and not flat2:
            return "ma"
        if flat2 and not flat1:
            return "am"
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

    def _canonical(self, atom: Atom) -> UniqueAtom:
        uatom: UniqueAtom = UniqueAtom(atom)
        return self.orig_atom.get(uatom, uatom)

    def _arg_contains_canonical_atom(self, arg: Hyperedge, target: UniqueAtom) -> bool:
        return any(self._canonical(a) == target for a in arg.all_atoms())

    @staticmethod
    def _is_plus_compound(edge: Hyperedge) -> bool:
        # +/B/. (and argrole'd variants like +/B.am/.) compound builders.
        # Treated as opaque by the deepen-modifier stage so modifiers
        # don't get pushed inside a compound concept.
        if edge.atom or not edge[0].atom:
            return False
        return cast(Atom, edge[0]).root() == "+"

    def _try_deepen_modifier(self, edge: Hyperedge) -> Hyperedge:
        # edge has the form (m/M E); push m/M into the deepest E-arg
        # justified by the DPT. Recurses so the modifier can cascade
        # through several levels in one go. Never swaps past another
        # modifier — if E is itself an M-edge, leave the structure alone.
        if edge[1].atom:
            return edge
        if edge[1].cmt == "M":
            return edge
        if self._is_plus_compound(edge[1]):
            return edge
        m_atom: Hyperedge = edge[0]
        inner: Hyperedge = edge[1]
        m_parent: UniqueAtom | None = self._dpt_parent_of.get(self._canonical(m_atom))
        if m_parent is None:
            return edge
        for i in range(1, len(inner)):
            arg: Hyperedge = inner[i]
            if self._arg_contains_canonical_atom(arg, m_parent):
                new_arg: Hyperedge = hedge([m_atom, arg])
                new_arg = self._try_deepen_modifier(new_arg)
                new_inner: list[Hyperedge] = list(inner)
                new_inner[i] = new_arg
                return hedge(new_inner)
        return edge

    def _deepen_modifiers(self, edge: Hyperedge) -> Hyperedge:
        if edge.atom:
            return edge
        if self._is_plus_compound(edge):
            return edge
        new_edge: Hyperedge | None = hedge(
            [self._deepen_modifiers(subedge) for subedge in edge]
        )
        if new_edge is None:
            return edge
        edge = new_edge
        if edge.cmt == "M" and len(edge) == 2 and not edge[1].atom:
            return self._try_deepen_modifier(edge)
        return edge

    def _update_atom(self, old: Atom, new: Atom) -> None:
        uold: Atom = UniqueAtom(old)
        unew: Atom = UniqueAtom(new)
        if uold in self.atom2token:
            self.atom2token[unew] = self.atom2token[uold]
        # Always store the canonical seed UniqueAtom — not the immediate
        # predecessor. Otherwise repeated argrole rewrites
        # (_apply_arg_roles → _fix_argroles → _replace_argroles → here)
        # build a chain where orig_atom maps to intermediate argroled
        # atoms whose `atom_obj` doesn't match any canonical in
        # `_dpt_directed`. The lock-in scan in `_distortion_delta` then
        # silently misses every relevant DPT pair → distortion always
        # comes out to 0.
        self.orig_atom[unew] = self.orig_atom.get(uold, uold)

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
            # Walk to the canonical so repeated argrole rewrites (this
            # function recurses into sub-edges, and edges built in
            # earlier reduction iterations may already be argroled) don't
            # leave a multi-step chain in orig_atom — every key must
            # map directly to the seed canonical, or the lock-in scan
            # in _distortion_delta will silently miss every DPT pair
            # touching that atom.
            self.orig_atom[unew_pred] = self.orig_atom.get(upred, upred)
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
                # Walk to the canonical so the chain stays one-deep.
                self.orig_atom[unew_builder] = self.orig_atom.get(ubuilder, ubuilder)
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

    def _build_seed_atomization(
        self,
        sentence: Span,
        prediction: tuple[tuple[str, ...] | list[str], list[list[tuple[str, float]]]]
        | None = None,
    ) -> tuple[list[Atom], list[AtomTrace], list[_AtomAlt]]:
        # Builds the all-top-1 atom sequence and pre-registers the
        # runner-up alternative for every uncertain token. Downstream
        # hill-climb code substitutes `seed_seq[alt.seq_pos] = alt.uatom`
        # without further state mutation; both primary and alt atoms are
        # already registered in self.atom2token / self.orig_atom.
        #
        # Returns:
        #   seed_seq: the top-1 atom sequence (UniqueAtoms),
        #   atom_traces: one per token, with chosen_label_rank=0 (caller
        #     bumps to 1 for any tok_idx that ends up substituted),
        #   alts: list of _AtomAlt for every uncertain token, sorted
        #     ascending by top-1 probability so iteration order doubles
        #     as the "lowest-confidence-first" tiebreak.
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

        # Eligibility: top-2 exists and has a different main type (first
        # character) from top-1. Same-main-type runner-ups (e.g. Cd vs Cx)
        # are just subtype refinements and stay fixed at top-1. The
        # uncertain_atom_ratio is then applied against the eligible pool;
        # the bottom int(ratio * len(eligible)) by top-1 probability
        # become uncertain.
        top1_scores: list[float] = [
            (cands[0][1] if cands else 1.0) for cands in top_candidates
        ]
        eligible_indices: list[int] = []
        for tok_idx, (top1_label, candidates) in enumerate(
            zip(atom_types, top_candidates, strict=True)
        ):
            if len(candidates) < 2 or not top1_label:
                continue
            top2_label: str = candidates[1][0]
            if not top2_label or top1_label[0] == top2_label[0]:
                continue
            eligible_indices.append(tok_idx)
        n_uncertain: int = int(self.uncertain_atom_ratio * len(eligible_indices))
        if n_uncertain > 0:
            eligible_indices.sort(key=lambda i: top1_scores[i])
            uncertain_indices: set[int] = set(eligible_indices[:n_uncertain])
        else:
            uncertain_indices = set()

        # Build the seed sequence (top-1 only) and the per-token traces.
        seed_seq: list[Atom] = []
        atom_traces: list[AtomTrace] = []
        primary_atoms_per_token: list[UniqueAtom | None] = [None] * len(atom_types)
        seq_pos_per_token: list[int] = [-1] * len(atom_types)

        for tok_idx, (token, label) in enumerate(
            zip(sentence, atom_types, strict=True)
        ):
            atom, refined_type = self._parse_token(token, label)
            if atom is not None:
                uatom: UniqueAtom = cast(UniqueAtom, UniqueAtom(atom))
                self.atom2token[uatom] = token
                self.token2atom[token] = uatom
                self.orig_atom[uatom] = uatom
                primary_atoms_per_token[tok_idx] = uatom
                seq_pos_per_token[tok_idx] = len(seed_seq)
                seed_seq.append(uatom)
            atom_traces.append(
                AtomTrace(
                    token_text=token.text,
                    token_idx=token.i,
                    predicted_type=label,
                    refined_type=refined_type,
                    final_atom=str(atom) if atom is not None else "",
                    dropped=atom is None,
                    top_candidates=top_candidates[tok_idx],
                    chosen_label_rank=0,
                    is_uncertain=tok_idx in uncertain_indices,
                )
            )

        # Pre-build and register every uncertain token's runner-up atom.
        # An alt that yields no atom (X-type) or whose primary was dropped
        # contributes nothing — those positions stay locked at their
        # primary label.
        alts: list[_AtomAlt] = []
        for tok_idx in uncertain_indices:
            primary = primary_atoms_per_token[tok_idx]
            if primary is None:
                continue
            candidates = top_candidates[tok_idx]
            if len(candidates) < 2:
                continue
            alt_label, _alt_score = candidates[1]
            if not alt_label or alt_label == atom_types[tok_idx]:
                continue
            token = sentence[tok_idx]
            alt_atom, alt_refined = self._parse_token(token, alt_label)
            if alt_atom is None:
                continue
            alt_uatom: UniqueAtom = cast(UniqueAtom, UniqueAtom(alt_atom))
            self.atom2token[alt_uatom] = token
            self.orig_atom[alt_uatom] = primary
            alts.append(
                _AtomAlt(
                    tok_idx=tok_idx,
                    seq_pos=seq_pos_per_token[tok_idx],
                    uatom=alt_uatom,
                    label=alt_label,
                    refined_type=alt_refined,
                    final_atom_str=str(alt_atom),
                    top1_prob=top1_scores[tok_idx],
                )
            )
        alts.sort(key=lambda a: a.top1_prob)

        self.debug_msg(f"Seed atom sequence: {seed_seq}")
        self.debug_msg(f"Uncertain alternatives: {[a.tok_idx for a in alts]}")
        return seed_seq, atom_traces, alts

    def _apply_forced_substitutions(
        self,
        sentence: Span,
        seed_seq: list[Atom],
        alts: list[_AtomAlt],
        atom_traces: list[AtomTrace],
        forced_substitutions: dict[int, str],
    ) -> None:
        # Patch the seed sequence at each (tok_idx, alt_label) position in
        # `forced_substitutions`. Prefer an existing pre-built _AtomAlt
        # (already registered in atom2token / orig_atom by
        # _build_seed_atomization) but fall through to a fresh build if
        # the requested label wasn't in the alts list — that can happen
        # when the user replays a trial whose tok_idx is no longer in
        # the uncertain pool (e.g. ratio changed) but the same sentence
        # still produces an atom for the requested label.
        alt_by_idx: dict[int, _AtomAlt] = {a.tok_idx: a for a in alts}
        for tok_idx, alt_label in forced_substitutions.items():
            if not 0 <= tok_idx < len(atom_traces):
                continue
            candidate = alt_by_idx.get(tok_idx)
            if candidate is not None and candidate.label == alt_label:
                alt = candidate
            else:
                token: Token = sentence[tok_idx]
                primary = self.token2atom.get(token)
                if primary is None:
                    # Primary token was dropped (X-type) — can't replay.
                    continue
                seq_pos = seed_seq.index(primary) if primary in seed_seq else -1
                if seq_pos < 0:
                    continue
                alt_atom, alt_refined = self._parse_token(token, alt_label)
                if alt_atom is None:
                    continue
                alt_uatom: UniqueAtom = cast(UniqueAtom, UniqueAtom(alt_atom))
                self.atom2token[alt_uatom] = token
                self.orig_atom[alt_uatom] = cast(UniqueAtom, primary)
                alt = _AtomAlt(
                    tok_idx=tok_idx,
                    seq_pos=seq_pos,
                    uatom=alt_uatom,
                    label=alt_label,
                    refined_type=alt_refined,
                    final_atom_str=str(alt_atom),
                    top1_prob=0.0,
                )
            seed_seq[alt.seq_pos] = alt.uatom
            atom_traces[tok_idx].chosen_label_rank = 1
            atom_traces[tok_idx].predicted_type = alt.label
            atom_traces[tok_idx].refined_type = alt.refined_type
            atom_traces[tok_idx].final_atom = alt.final_atom_str

    def _compute_depths_and_connections(self, root: Token, depth: int = 0) -> None:
        # Populates self.connections with directional (parent_atom,
        # child_atom) pairs from the spaCy dep tree (root descends into
        # its children). Only one direction is stored — the spaCy tree
        # is the source of truth for dep direction.
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
            self._compute_depths_and_connections(child, depth + 1)

    def _compute_dpt_pairs(self) -> None:
        # Canonicalise the dep parent-child edges in self.connections
        # into:
        # - self._dpt_pairs: undirected UniqueAtom pairs, used for the
        #   undirected satisfaction check (most SH rules);
        # - self._dpt_directed: ordered (parent, child) UniqueAtom
        #   tuples, used for the strict-direction check on default-case
        #   SH pairs (rule 3.1 / 3.9 — connector_head -> arg_head).
        self._dpt_pairs = set()
        self._dpt_directed = set()
        self._dpt_parent_of = {}
        self._dpt_children_of = {}
        for a, b in self.connections:
            ua_parent = self.orig_atom.get(a)
            ua_child = self.orig_atom.get(b)
            if ua_parent is None or ua_child is None or ua_parent == ua_child:
                continue
            self._dpt_pairs.add(frozenset({ua_parent, ua_child}))
            self._dpt_directed.add((ua_parent, ua_child))
            self._dpt_parent_of[ua_child] = ua_parent
            self._dpt_children_of.setdefault(ua_parent, set()).add(ua_child)

    def _compute_dpt_separators(self, root: Token) -> None:
        # Criterion 3.12 precomputation. For each spaCy token T whose
        # canonical atom is in the DPT, walk T's spaCy children in
        # order and split that sequence on X-labeled (dropped) and
        # J-labeled (conjunction) children — each such child is a
        # *separator*. For every separator we record (left_set,
        # right_set), where each side is the union of canonical atoms
        # in the subtrees of the non-separator children on that side
        # of the separator; the parent atom is added to left_set
        # ("parent belongs to the first separated group"). Separator
        # children's own subtrees contribute to neither side.
        #
        # Walks spaCy directly (not the DPT) because X-tokens are
        # dropped from token2atom and would otherwise be invisible.
        self._dpt_separators = {}

        def is_sep(tok: Token) -> bool:
            # X (dropped): not in token2atom.
            if tok not in self.token2atom:
                return True
            ua = self.orig_atom.get(self.token2atom[tok])
            # J (conjunction): mtype == "J".
            return ua is not None and ua.mtype() == "J"

        def subtree_atoms(tok: Token) -> set[UniqueAtom]:
            out: set[UniqueAtom] = set()
            for d in tok.subtree:
                a = self.token2atom.get(d)
                if a is None:
                    continue
                ua = self.orig_atom.get(a)
                if ua is not None:
                    out.add(ua)
            return out

        for tok in root.subtree:
            a = self.token2atom.get(tok)
            if a is None:
                continue
            parent_ua = self.orig_atom.get(a)
            if parent_ua is None:
                continue
            children: list[Token] = list(tok.children)
            sep_positions: list[int] = [i for i, c in enumerate(children) if is_sep(c)]
            if not sep_positions:
                continue
            child_sub: list[frozenset[UniqueAtom]] = [
                frozenset() if is_sep(c) else frozenset(subtree_atoms(c))
                for c in children
            ]
            pairs: list[tuple[frozenset[UniqueAtom], frozenset[UniqueAtom]]] = []
            for i in sep_positions:
                left: set[UniqueAtom] = {parent_ua}
                for j in range(i):
                    left |= child_sub[j]
                right: set[UniqueAtom] = set()
                for j in range(i + 1, len(children)):
                    right |= child_sub[j]
                # No atoms on either side → no candidate can ever
                # straddle this separator. Skip to keep the per-edge
                # scan in _distortion_delta lean.
                if not right:
                    continue
                if len(left) == 1 and parent_ua in left:
                    # Only the parent on the left — a leading separator.
                    # A candidate that places the parent and any right-
                    # side atom in different children of a new edge
                    # would still straddle, so we keep the split.
                    pass
                pairs.append((frozenset(left), frozenset(right)))
            if pairs:
                self._dpt_separators[parent_ua] = pairs

    def _score(
        self, edges: list[Hyperedge], new_edge: Hyperedge, parent_pos: int
    ) -> tuple[int, bool]:
        # Candidates are dep-connected by construction (parent + dep-children),
        # so the connectivity term is gone. The score is the non-dangling
        # bonus plus a depth term (mdepth * 100) that rewards bottom-up
        # reductions — the candidate's shallowest atom should sit deep in
        # the dep tree — and an atom-count tiebreaker. Returns (score,
        # no_dangling) so callers can use the no_dangling flag for the
        # dominance filter without recomputing _no_dangling_branches.
        no_dangling: bool = self._no_dangling_branches(edges, new_edge, parent_pos)
        mdepth: int = 9999999
        n: int = 0
        for edge in edges:
            for atom in edge.all_atoms():
                if atom in self.orig_atom:
                    oatom: Atom = self.orig_atom[atom]
                    if oatom in self.depths:
                        n += 1
                        depth: int = self.depths[oatom]
                        if depth < mdepth:
                            mdepth = depth
        score: int = (10000000 if no_dangling else 0) + (mdepth * 100) + n
        return score, no_dangling

    @staticmethod
    def _is_special_transparent_atom(atom: Atom) -> bool:
        # System atoms whose namespace is "." (e.g. +/B/., :/J/.,
        # poss/Bp.am/., list/J/.). Per the distortion rules, these
        # connectors are fully transparent: they introduce no SH
        # parent-child edges, and their containing edge passes through
        # to the args' heads.
        parts: list[str] = atom.parts()
        return len(parts) >= 3 and parts[-1] == "."

    @staticmethod
    def _builder_main_aux_indices(argroles: str) -> tuple[int, int] | None:
        # Position of the "m" (main) and "a" (auxiliary) chars in a
        # builder's argroles. Returns None if either is absent.
        m_idx: int = argroles.find("m")
        a_idx: int = argroles.find("a")
        if m_idx < 0 or a_idx < 0:
            return None
        return m_idx, a_idx

    def _sh_head_atoms(self, edge: Hyperedge) -> set[UniqueAtom]:
        # Canonical atoms that represent `edge` externally — used by
        # enclosing edges to identify the "anchor" of `edge` for
        # parent-child purposes. Modifiers, special transparents, and
        # builders all delegate the head downward per the rules.
        if edge.atom:
            ua = self.orig_atom.get(cast(Atom, edge))
            return {ua} if ua is not None else set()

        conn_atom = edge.connector_atom()
        if conn_atom is None:
            return set()

        if self._is_special_transparent_atom(conn_atom):
            # Rule 3.10: a special-transparent builder with m/a argroles
            # (e.g. (+/B.am x/C y/C) or (poss/Bp.am/. ...)) is still
            # transparent to its main concept like any builder — the head
            # comes from the main arg, not from a union of arg heads.
            if conn_atom.mtype() == "B":
                indices = self._builder_main_aux_indices(edge.argroles())
                args_list: list[Hyperedge] = list(edge[1:])
                if indices is not None and indices[0] < len(args_list):
                    return self._sh_head_atoms(args_list[indices[0]])
            # Otherwise (bare +/B/. with no argroles, :/J/., list/J/.,
            # etc.): fully transparent, head = union of arg heads.
            out: set[UniqueAtom] = set()
            for arg in edge[1:]:
                out |= self._sh_head_atoms(arg)
            return out

        cmt: str = conn_atom.mtype()

        if cmt == "M":
            # Modifier construction: head propagates from the (single) arg.
            if len(edge) > 1:
                return self._sh_head_atoms(edge[1])
            return set()

        if cmt == "B":
            # Non-special builder: head is the main concept.
            ar: str = edge.argroles()
            args_list: list[Hyperedge] = list(edge[1:])
            indices = self._builder_main_aux_indices(ar)
            if indices is not None and indices[0] < len(args_list):
                return self._sh_head_atoms(args_list[indices[0]])
            # Argroles missing "m"/"a": fall back to first arg as head.
            if args_list:
                return self._sh_head_atoms(args_list[0])
            return set()

        # Default (P, T, J non-special, etc.): head = the connector's
        # innermost atom.
        ua = self.orig_atom.get(conn_atom)
        return {ua} if ua is not None else set()

    def _sh_local_pairs(
        self, edge: Hyperedge
    ) -> tuple[
        set[frozenset[UniqueAtom]],
        set[tuple[UniqueAtom, UniqueAtom]],
    ]:
        # SH parent-child pairs introduced by `edge`'s own local
        # structure. Returns a (undirected, directional) tuple:
        # - undirected: pairs that match a DPT edge regardless of which
        #   side is parent (rules 3.4-3.7, 3.10, and the modifier chain
        #   walk). Stored as frozenset({a, b}).
        # - directional: ordered (parent, child) tuples that must match
        #   DPT direction exactly (rule 3.1 / 3.9 — connector_head as
        #   parent of arg_head in P/T/J non-special edges).
        #
        # Recursion into the connector chain is included so modifier
        # chain edges are captured here, but recursion into the args is
        # not — those are charged when their own subtrees were built.
        undirected: set[frozenset[UniqueAtom]] = set()
        directional: set[tuple[UniqueAtom, UniqueAtom]] = set()
        if edge.atom:
            return undirected, directional

        conn_atom = edge.connector_atom()
        if conn_atom is None:
            return undirected, directional

        # Special transparent connector: usually contributes nothing
        # (rule 3.8). Exception (rule 3.10): a special-transparent
        # builder with m/a argroles (e.g. (+/B.am x/C y/C)) emits the
        # single undirected {head(main), head(aux)} pair — the
        # structure is transparent to its main concept like any
        # builder, but the builder atom itself has no DPT token, so we
        # never emit pairs involving it.
        if self._is_special_transparent_atom(conn_atom):
            if conn_atom.mtype() == "B":
                indices = self._builder_main_aux_indices(edge.argroles())
                special_args: list[Hyperedge] = list(edge[1:])
                if (
                    indices is not None
                    and len(special_args) >= 2
                    and indices[0] < len(special_args)
                    and indices[1] < len(special_args)
                    and indices[0] != indices[1]
                ):
                    main_heads_s = self._sh_head_atoms(special_args[indices[0]])
                    aux_heads_s = self._sh_head_atoms(special_args[indices[1]])
                    for mh in main_heads_s:
                        for ah in aux_heads_s:
                            if mh != ah:
                                undirected.add(frozenset({mh, ah}))
            return undirected, directional

        # Walk a non-atomic connector chain (modifier-wrapped connector
        # like (will/Mm be/Pv.so) or ((very/Ma smart/Ma) student/...))
        # and emit modifier-target edges along the way. These are
        # modifier-rule pairs (3.4 / 3.6) — always undirected.
        connector: Hyperedge = edge[0]
        if connector.not_atom:
            chain: Hyperedge = connector
            while chain.not_atom and len(chain) >= 2:
                mod: Hyperedge = chain[0]
                target: Hyperedge = chain[1]
                if not mod.atom:
                    break
                target_atom_obj: Atom = (
                    target.inner_atom() if target.not_atom else cast(Atom, target)
                )
                ua_mod = self.orig_atom.get(cast(Atom, mod))
                ua_t = self.orig_atom.get(target_atom_obj)
                if ua_mod is not None and ua_t is not None and ua_mod != ua_t:
                    undirected.add(frozenset({ua_mod, ua_t}))
                chain = target

        cmt: str = conn_atom.mtype()
        args: list[Hyperedge] = list(edge[1:])

        if cmt == "M":
            # (M, X): M considers head(X) its child. Asymmetric per rule
            # 3.4, but we store undirected pairs so the asymmetry doesn't
            # affect satisfaction.
            ua_m = self.orig_atom.get(conn_atom)
            if ua_m is not None:
                for arg in args:
                    for h in self._sh_head_atoms(arg):
                        if h != ua_m:
                            undirected.add(frozenset({ua_m, h}))
            return undirected, directional

        if cmt == "B":
            # Non-special builder: rule 3.7 accepts either of two
            # configs (main → aux → builder, or main → builder → aux),
            # so the union is the full undirected triangle on
            # {builder_atom, head(main), head(aux)}.
            ar: str = edge.argroles()
            indices = self._builder_main_aux_indices(ar)
            ua_b = self.orig_atom.get(conn_atom)
            if indices is not None and ua_b is not None and len(args) >= 2:
                m_idx, a_idx = indices
                if m_idx < len(args) and a_idx < len(args) and m_idx != a_idx:
                    main_heads = self._sh_head_atoms(args[m_idx])
                    aux_heads = self._sh_head_atoms(args[a_idx])
                    for mh in main_heads:
                        for ah in aux_heads:
                            if mh != ah:
                                undirected.add(frozenset({mh, ah}))
                            if mh != ua_b:
                                undirected.add(frozenset({mh, ua_b}))
                            if ah != ua_b:
                                undirected.add(frozenset({ah, ua_b}))
                    return undirected, directional
            # Fall through to the default rule when argroles lack a
            # usable m/a pair (defensive — argroles should be set by
            # _apply_arg_roles before we get here).

        # Default (P, T, J non-special, etc.) — rule 3.1 / 3.9:
        # connector head is parent of head(arg_i) for each i, and the
        # DPT direction must match (parent token on DPT must equal the
        # connector head token, no inversion).
        ua_conn = self.orig_atom.get(conn_atom)
        if ua_conn is not None:
            for arg in args:
                for h in self._sh_head_atoms(arg):
                    if h != ua_conn:
                        directional.add((ua_conn, h))
        return undirected, directional

    def _dep_shallowest_atoms(self, edge: Hyperedge) -> set[UniqueAtom]:
        # Atoms (mapped through orig_atom) of tokens in `edge` whose spaCy
        # dep-parent token sits OUTSIDE `edge`. Same notion of "shallowest"
        # used in _no_dangling_branches, but applied to a hyperedge
        # subtree. Used by the relaxed parent/child satisfaction check in
        # _distortion_delta to broaden what counts as a slot's head.
        ua_by_token: dict[Token, UniqueAtom] = {}
        for a in edge.all_atoms():
            ua: UniqueAtom | None = self.orig_atom.get(a)
            if ua is None:
                continue
            tok: Token | None = self.atom2token.get(ua)
            if tok is not None:
                ua_by_token[tok] = ua
        tokens: set[Token] = set(ua_by_token)
        result: set[UniqueAtom] = set()
        for tok, ua in ua_by_token.items():
            if tok.head == tok or tok.head not in tokens:
                result.add(ua)
        return result

    def _j_3_7_1_satisfied(
        self,
        conn_atom: Atom,
        child_atoms: list[set[UniqueAtom]],
    ) -> bool:
        # Rule 3.7.1: a non-special J-edge contributes 0 distortion iff
        # its J-connector atom has, in the DPT,
        #   (a) a parent-child relationship (either direction) with at
        #       least one atom inside one of the args (child_atoms[i],
        #       i >= 1), and
        #   (b) a parent-child or sibling relationship with at least one
        #       atom inside a *different* arg.
        ua_j = self.orig_atom.get(conn_atom)
        if ua_j is None:
            return False
        j_parent: UniqueAtom | None = self._dpt_parent_of.get(ua_j)
        j_children: set[UniqueAtom] = self._dpt_children_of.get(ua_j, set())
        j_siblings: set[UniqueAtom] = (
            (self._dpt_children_of.get(j_parent, set()) - {ua_j})
            if j_parent is not None
            else set()
        )
        a_pc: set[int] = set()
        a_pc_or_sib: set[int] = set()
        for idx in range(1, len(child_atoms)):
            atoms = child_atoms[idx]
            if not atoms:
                continue
            pc: bool = (j_parent is not None and j_parent in atoms) or bool(
                atoms & j_children
            )
            if pc:
                a_pc.add(idx)
                a_pc_or_sib.add(idx)
                continue
            if atoms & j_siblings:
                a_pc_or_sib.add(idx)
        return bool(a_pc) and len(a_pc_or_sib) >= 2

    def _distortion_delta(self, new_edge: Hyperedge, relaxed: bool = False) -> int:
        # Charge +1 for each DPT parent-child pair that "locks in" at
        # this edge (i.e. its endpoints first co-occur here, sitting in
        # distinct immediate children of new_edge) and is *not*
        # satisfied by an SH parent-child edge introduced by this
        # edge's local structure.
        #
        # Satisfaction has two modes:
        # - An undirected SH pair (modifier, builder, special-builder,
        #   modifier-chain) matches any DPT pair between the same atoms
        #   regardless of direction.
        # - A directional SH pair (parent, child) from the default
        #   P/T/J case (rule 3.1 / 3.9) only matches a DPT edge with
        #   the SAME direction — connector_head must be the DPT parent.
        #
        # `relaxed=True` (opt-in per rule, e.g. the `:/J/.` rule) adds a
        # third route: a DPT pair is also satisfied if its parent atom is
        # in the slot's broadened head set and its child atom is in the
        # other slot's broadened head set, where the broadened head set
        # is _sh_head_atoms(child_slot) | _dep_shallowest_atoms(child_slot).
        # This rescues pairs that lock in at an edge whose connector
        # itself emits no local SH pair (e.g. :/J/.).
        #
        # Once a pair locks in, no future construction can introduce an
        # SH edge directly between its atoms — outer constructions only
        # connect the surviving heads, not buried atoms. Charging at
        # lock-in is therefore final, and the cumulative total is
        # monotone non-decreasing.
        if new_edge.atom:
            return 0
        children: list[Hyperedge] = list(new_edge)
        if len(children) < 2:
            return 0

        child_atoms: list[set[UniqueAtom]] = []
        for c in children:
            s: set[UniqueAtom] = set()
            for a in c.all_atoms():
                ua = self.orig_atom.get(a)
                if ua is not None:
                    s.add(ua)
            child_atoms.append(s)

        # Rule 3.7.1 — non-special J-connector special case.
        # If the J-connector atom has a DPT parent-child relationship with
        # any atom in one arg, AND a parent-child or sibling relationship
        # with any atom in a different arg, the J-edge is considered fully
        # satisfied (0 distortion). Otherwise it emits no local SH pairs:
        # every DPT pair that locks in here will charge unless rescued by
        # the relaxed head route.
        conn_atom = new_edge.connector_atom()
        non_special_j: bool = (
            conn_atom is not None
            and conn_atom.mtype() == "J"
            and not self._is_special_transparent_atom(conn_atom)
        )
        if non_special_j:
            if self._j_3_7_1_satisfied(conn_atom, child_atoms):
                return 0
            local_undirected: set[frozenset[UniqueAtom]] = set()
            local_directional: set[tuple[UniqueAtom, UniqueAtom]] = set()
        else:
            local_undirected, local_directional = self._sh_local_pairs(new_edge)

        # Lazy per-slot broadened head set (SH-resolved | dep-shallowest),
        # only computed when the relaxed route is in play.
        head_sets: list[set[UniqueAtom] | None] = [None] * len(children)

        def _head_set(idx: int) -> set[UniqueAtom]:
            cached = head_sets[idx]
            if cached is not None:
                return cached
            hs: set[UniqueAtom] = self._sh_head_atoms(
                children[idx]
            ) | self._dep_shallowest_atoms(children[idx])
            head_sets[idx] = hs
            return hs

        distortion: int = 0
        for ua_parent, ua_child in self._dpt_directed:
            i_p: int = -1
            i_c: int = -1
            for idx, atoms in enumerate(child_atoms):
                if i_p < 0 and ua_parent in atoms:
                    i_p = idx
                if i_c < 0 and ua_child in atoms:
                    i_c = idx
                if i_p >= 0 and i_c >= 0:
                    break
            if i_p < 0 or i_c < 0 or i_p == i_c:
                continue
            if (ua_parent, ua_child) in local_directional:
                continue
            if frozenset({ua_parent, ua_child}) in local_undirected:
                continue
            if relaxed and ua_parent in _head_set(i_p) and ua_child in _head_set(i_c):
                continue
            distortion += 1

        # Criterion 3.12 — X/J separator straddling lock-in.
        # For each precomputed (left_set, right_set) split at some DPT
        # parent's children level, locate the first immediate child of
        # new_edge whose atoms hit left_set and the first that hits
        # right_set. If both are present in *different* immediate
        # children, this edge is the lock-in for that separator (no
        # smaller edge already contained both sides in its own
        # immediate children). Charge +1 iff the parent's side
        # (left_set, which always contains the parent atom) has
        # remainder atoms not present anywhere in new_edge's atoms.
        # Charging only on left-side remainder matches the user's
        # rule: the parent belongs to the first separated group, and
        # the criterion penalises a candidate that reaches across the
        # separator while leaving the parent's own group incomplete.
        # Right-side remainder is fine — it just means the candidate
        # is building the parent's argument list incrementally and
        # the still-unattached right-side sibling will be folded in
        # by a later edge.
        if self._dpt_separators:
            new_edge_atoms: set[UniqueAtom] = set()
            for s in child_atoms:
                new_edge_atoms |= s
            for sep_list in self._dpt_separators.values():
                for left_set, right_set in sep_list:
                    if not (left_set - new_edge_atoms):
                        continue
                    i_left: int = -1
                    i_right: int = -1
                    for idx, atoms in enumerate(child_atoms):
                        if i_left < 0 and atoms & left_set:
                            i_left = idx
                        if i_right < 0 and atoms & right_set:
                            i_right = idx
                        if i_left >= 0 and i_right >= 0:
                            break
                    if i_left < 0 or i_right < 0 or i_left == i_right:
                        continue
                    distortion += 1
        return distortion

    def diagnose_distortion(self, edge: Hyperedge) -> list[str]:
        # Walk `edge` post-order and report, per non-atomic sub-edge,
        # which DPT pairs lock in there and whether they're satisfied.
        # Reads self._dpt_directed and self.orig_atom from the most
        # recent parse_spacy_sentence call — re-parse the same sentence
        # before calling this if state could be stale.
        lines: list[str] = []
        seen: set[int] = set()
        self._diagnose_distortion_recurse(edge, lines, seen)
        return lines

    def _diagnose_distortion_recurse(
        self, edge: Hyperedge, lines: list[str], seen: set[int]
    ) -> None:
        if edge.atom:
            return
        for child in edge:
            self._diagnose_distortion_recurse(child, lines, seen)
        edge_id: int = id(edge)
        if edge_id in seen:
            return
        seen.add(edge_id)
        self._diagnose_one_edge(edge, lines)

    def _diagnose_one_edge(self, edge: Hyperedge, lines: list[str]) -> None:
        children: list[Hyperedge] = list(edge)
        if len(children) < 2:
            return

        child_atoms: list[set[UniqueAtom]] = []
        for c in children:
            s: set[UniqueAtom] = set()
            for a in c.all_atoms():
                ua = self.orig_atom.get(a)
                if ua is not None:
                    s.add(ua)
            child_atoms.append(s)

        local_undirected, local_directional = self._sh_local_pairs(edge)

        successes: list[str] = []
        failures: list[str] = []
        for ua_parent, ua_child in sorted(
            self._dpt_directed, key=lambda p: (str(p[0]), str(p[1]))
        ):
            i_p: int = -1
            i_c: int = -1
            for idx, atoms in enumerate(child_atoms):
                if i_p < 0 and ua_parent in atoms:
                    i_p = idx
                if i_c < 0 and ua_child in atoms:
                    i_c = idx
                if i_p >= 0 and i_c >= 0:
                    break
            if i_p < 0 or i_c < 0 or i_p == i_c:
                continue
            pair_str = f"({ua_parent} → {ua_child})"
            if (ua_parent, ua_child) in local_directional:
                successes.append(f"{pair_str} directional ✓")
            elif frozenset({ua_parent, ua_child}) in local_undirected:
                successes.append(f"{pair_str} undirected ✓")
            else:
                failures.append(pair_str)

        if not (successes or failures):
            return

        lines.append(f"Edge: {edge}")
        if local_directional:
            dir_pairs = sorted(f"({p} → {c})" for p, c in local_directional)
            lines.append(f"  SH directional: {dir_pairs}")
        if local_undirected:
            und_pairs = sorted(
                "{" + ", ".join(sorted(str(a) for a in p)) + "}"
                for p in local_undirected
            )
            lines.append(f"  SH undirected: {und_pairs}")
        for s in successes:
            lines.append(f"  ✓ {s}")
        for f in failures:
            lines.append(f"  ✗ {f}  (UNSATISFIED, +1 distortion)")
        lines.append("")

    def _candidate_dominance_atoms(self, edge: Hyperedge) -> frozenset:
        # Each atom contributes by atom_str with the argroles subpart
        # stripped, so two argrole variants of the same connector collapse
        # to a single signature element — otherwise the strict-subset
        # dominance check would never fire across candidates whose argroles
        # differ. We can't use orig_atom here: UniqueAtom hashes by
        # id(atom_obj), and the atoms in `edge` were freshly wrapped after
        # _apply_arg_roles, so they never match keys registered earlier.
        # Rule-introduced specials (e.g. '+/B/.', ':/J/.') normalize the
        # same way ('+/B.am/.' -> '+/B/.'), so a parse carrying an extra
        # connector still differs from one that lacks it but is invariant
        # to its argrole binding.
        sig: list[str] = []
        for a in edge.all_atoms():
            atom_str: str = a.atom_str
            parts: list[str] = atom_str.split("/")
            if len(parts) >= 2:
                role: list[str] = parts[1].split(".")
                if len(role) >= 2:
                    parts[1] = role[0]
                    atom_str = "/".join(parts)
            sig.append(atom_str)
        return frozenset(sig)

    def _plus_builder_bonus(self, edge: Hyperedge) -> int:
        # Reward +/B/. builder candidates whose token-derived atoms cover a
        # contiguous span of the source sentence — this biases the search
        # toward compounds that respect surface adjacency over long-range
        # stitches. The connector may carry argroles after _apply_arg_roles
        # (e.g. "+/B.ma/."), but its root remains "+".
        if edge.atom or not edge[0].atom:
            return 0
        if cast(Atom, edge[0]).root() != "+":
            return 0
        positions: set[int] = set()
        for atom in edge.all_atoms():
            token: Token | None = self.atom2token.get(atom)
            if token is not None:
                positions.add(token.i)
        if not positions:
            return 0
        if max(positions) - min(positions) + 1 != len(positions):
            return 0
        return 10 * len(positions)

    def _no_dangling_branches(
        self,
        window: list[Hyperedge],
        new_edge: Hyperedge,
        parent_pos: int,
    ) -> bool:
        # A "dangling root" x is a sentence token outside the candidate
        # window whose dep-parent is inside it. The candidate is acceptable
        # iff every such x can be absorbed by *some* rule whose inputs
        # include both the new candidate edge and the live current sequence
        # edge at x's position, and whose application produces an output
        # edge with both _candidate_badness == 0 and _distortion_delta == 0.
        # Extra slots, if the rule has size > 2, may be filled from any
        # other live edge in the post-application sequence.
        cur_seq: list[Hyperedge] | None = self._cur_sequence
        if cur_seq is None:
            return True

        cand_tokens: set[Token] = set()
        for edge in window:
            for atom in edge.all_atoms():
                tok: Token | None = self.atom2token.get(atom)
                if tok is not None:
                    cand_tokens.add(tok)
        if not cand_tokens:
            return True

        tok2pos: dict[Token, int] = {}
        for pos, edge in enumerate(cur_seq):
            for atom in edge.all_atoms():
                tok = self.atom2token.get(atom)
                if tok is not None:
                    tok2pos[tok] = pos

        cand_positions: set[int] = {
            pos for pos, edge in enumerate(cur_seq) if any(edge is w for w in window)
        }

        # Post-application "virtual" sequence: drop every window position,
        # splice new_edge in at parent_pos, preserve remaining order.
        virtual: list[Hyperedge] = []
        new_edge_v: int = -1
        orig_to_virtual: dict[int, int] = {}
        for pos, edge in enumerate(cur_seq):
            if pos == parent_pos:
                new_edge_v = len(virtual)
                orig_to_virtual[pos] = new_edge_v
                virtual.append(new_edge)
            elif pos in cand_positions:
                continue
            else:
                orig_to_virtual[pos] = len(virtual)
                virtual.append(edge)
        if new_edge_v < 0:
            return True

        any_tok: Token = next(iter(cand_tokens))
        sent: Span = any_tok.sent

        for x in sent:
            if x in cand_tokens:
                continue
            if x.head == x:
                continue
            if x.head not in cand_tokens:
                continue
            x_pos: int | None = tok2pos.get(x)
            if x_pos is None:
                continue
            if x_pos in cand_positions:
                continue
            x_v: int | None = orig_to_virtual.get(x_pos)
            if x_v is None:
                continue
            if not self._dangling_absorbable(virtual, new_edge_v, x_v):
                return False
        return True

    def _dangling_absorbable(
        self,
        virtual: list[Hyperedge],
        new_edge_v: int,
        x_v: int,
    ) -> bool:
        # True iff some rule fires on an input set that includes both
        # virtual[new_edge_v] and virtual[x_v] (plus 0..size-2 other live
        # positions from `virtual`) and yields an output edge with both
        # _candidate_badness == 0 and _distortion_delta == 0.
        others: list[int] = [
            v for v in range(len(virtual)) if v != new_edge_v and v != x_v
        ]
        # Try small rules first so easy absorptions short-circuit before
        # we enumerate combinatorial extras for the larger P/B rules.
        for rule in sorted(RULES, key=lambda r: r.size):
            need_extra: int = rule.size - 2
            if need_extra < 0 or need_extra > len(others):
                continue
            combos: Any
            if need_extra == 0:
                combos = ((),)
            else:
                combos = itertools.combinations(others, need_extra)
            for extras in combos:
                indices: list[int] = sorted({new_edge_v, x_v, *extras})
                if len(indices) != rule.size:
                    continue
                result = apply_rule_indices(rule, virtual, indices, self.atom2token)
                if result is None:
                    continue
                argroled = unique(
                    self._fix_argroles(self._apply_arg_roles(non_unique(result)))
                )
                if self._candidate_badness(argroled) != 0:
                    continue
                if (
                    self._distortion_delta(
                        argroled, relaxed=rule.relaxed_head_satisfaction
                    )
                    != 0
                ):
                    continue
                return True
        return False

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
        total: int = 0
        try:
            for issues in edge.check_correctness().values():
                for _err_type, _msg in issues:
                    total += _BADNESS_WEIGHTS[0]
            for issues in check_structural_quality(edge).values():
                for _err_type, _msg, severity in issues:
                    total += _BADNESS_WEIGHTS.get(severity, 0)
        except Exception:
            return _BADNESS_CRASH_PENALTY
        return total

    def _expand_state(self, state: _ParseState) -> _ParseState | None:
        # Generate every rule application that can fire on `state.sequence`,
        # run the dominance filter, and return the single best successor
        # state by (badness ASC, distortion ASC, score DESC). When no
        # rule applies, a `:/J/.` force-join fallback is produced with
        # `failed=True` so the caller's reduction loop still terminates.
        # Returns None only on an unrecoverable internal error.
        self._cur_sequence = state.sequence
        base_iter: RuleIteration = RuleIteration(
            iteration=len(state.iterations),
            sequence_repr=[str(e) for e in state.sequence],
        )
        # Parallel lists kept in lockstep until the dominance filter.
        # Per-candidate tuple:
        # (score, new_edge, indices, badness, distortion, parent_pos)
        # `indices` holds the positions in state.sequence consumed by the
        # rule, sorted ascending by sentence position. `parent_pos` is the
        # position of the dep-tree-shallowest hyperedge among indices — the
        # new edge replaces state.sequence[parent_pos] and the other
        # positions are dropped.
        cand_actions: list[tuple[int, Hyperedge, tuple[int, ...], int, int, int]] = []
        cand_records: list[RuleCandidate] = []
        cand_signatures: list[frozenset] = []
        cand_can_dominate: list[bool] = []
        cand_no_dangling: list[bool] = []
        cand_mandatory: list[bool] = []

        # token -> sequence index of the live hyperedge containing that token
        tok2pos: dict[Token, int] = {}
        for pos, edge in enumerate(state.sequence):
            for atom in edge.all_atoms():
                tok = self.atom2token.get(atom)
                if tok is not None:
                    tok2pos[tok] = pos

        def _live_dep_children(pos: int) -> set[int]:
            # Live child-hyperedge positions: every dep-child of any token
            # inside state.sequence[pos] that currently lives in a *different*
            # sequence position.
            out: set[int] = set()
            edge_tokens: set[Token] = {
                self.atom2token[a]
                for a in state.sequence[pos].all_atoms()
                if a in self.atom2token
            }
            for t in edge_tokens:
                for child in t.children:
                    cpos = tok2pos.get(child)
                    if cpos is not None and cpos != pos:
                        out.add(cpos)
            return out

        def _try_candidate(
            rule_number: int,
            rule: Rule,
            indices: tuple[int, ...],
            parent_pos: int,
        ) -> bool:
            new_edge: Hyperedge | None = apply_rule_indices(
                rule, state.sequence, list(indices), self.atom2token
            )
            if not new_edge or not self._is_relcl_constraint_satisfied(new_edge):
                return False
            # Assign argroles to the candidate before badness so the
            # P/B argrole-aware checks see realistic connectors, then
            # resolve the '?' placeholders _apply_arg_roles leaves
            # behind on relations whose dep-tag didn't map to a
            # role. The argroled edge propagates into state.sequence,
            # so later rule applications nest already-argroled
            # subedges (and the post-loop pass becomes idempotent).
            # Both helpers index self.atom2token by the bare Atom
            # identity stashed at parse_token time, so we have to
            # non_unique → apply → unique to keep state.sequence
            # uniformly UniqueAtom-wrapped (the rest of _expand_state
            # depends on that wrapping for orig_atom lookups).
            new_edge = unique(
                self._fix_argroles(self._apply_arg_roles(non_unique(new_edge)))
            )
            window: list[Hyperedge] = [state.sequence[k] for k in indices]
            base_score, no_dangling = self._score(window, new_edge, parent_pos)
            score: int = base_score + self._plus_builder_bonus(new_edge)
            bad: int = self._candidate_badness(new_edge)
            distortion: int = self._distortion_delta(
                new_edge, relaxed=rule.relaxed_head_satisfaction
            )
            distortion += _sibling_gap_distortion(indices)
            cand_records.append(
                RuleCandidate(
                    rule_index=rule_number,
                    rule_repr=rule_repr(rule, rule_number),
                    pos=parent_pos,
                    score=score,
                    new_edge_repr=str(new_edge),
                    badness=bad,
                    distortion=distortion,
                    indices=list(indices),
                )
            )
            cand_actions.append((score, new_edge, indices, bad, distortion, parent_pos))
            cand_signatures.append(self._candidate_dominance_atoms(new_edge))
            cand_can_dominate.append(rule.can_dominate)
            cand_no_dangling.append(no_dangling)
            cand_mandatory.append(rule.mandatory)
            return True

        # Dep-tree DFS candidate generation. For every live hyperedge H,
        # enumerate every connected dep-subtree of live edges rooted at H
        # (H = dep-shallowest live edge in the subtree). A candidate set
        # need not be limited to (parent + immediate children): nested
        # paths like {grandparent, parent, child} are valid as long as
        # they form a connected subtree. The "rooted at H" anchoring
        # ensures each subtree is enumerated exactly once.
        max_rule_size: int = max(r.size for r in RULES)

        children_of: dict[int, list[int]] = {
            pos: sorted(_live_dep_children(pos)) for pos in range(len(state.sequence))
        }

        # Per-position DPT-parent coverage. A position "covers" a DPT
        # parent atom P if any atom inside it is a direct dep-child of P.
        # dpt_sibling_positions[P] is the sorted list of all positions
        # that cover P — these are the DPT-siblings under P as seen from
        # state.sequence. Both dicts feed the sibling-gap distortion charge
        # applied per candidate in _sibling_gap_distortion below.
        dpt_parents_at_pos: dict[int, set[UniqueAtom]] = {}
        dpt_sibling_positions: dict[UniqueAtom, list[int]] = {}
        for pos in range(len(state.sequence)):
            parents_here: set[UniqueAtom] = set()
            for a in state.sequence[pos].all_atoms():
                ua = self.orig_atom.get(a)
                if ua is None:
                    continue
                p = self._dpt_parent_of.get(ua)
                if p is not None:
                    parents_here.add(p)
            dpt_parents_at_pos[pos] = parents_here
            for p in parents_here:
                dpt_sibling_positions.setdefault(p, []).append(pos)

        def _sibling_gap_distortion(indices: tuple[int, ...]) -> int:
            # +1 per non-selected interleaving DPT-sibling. For each DPT
            # parent that has >= 2 selected positions, sort the selected
            # positions by state.sequence index and count, between each
            # successive selected pair, how many positions that *also*
            # cover the same parent fall strictly between them and aren't
            # themselves selected.
            selected: set[int] = set(indices)
            parents_with_selected: dict[UniqueAtom, list[int]] = {}
            for i in indices:
                for p in dpt_parents_at_pos.get(i, ()):
                    parents_with_selected.setdefault(p, []).append(i)
            extra: int = 0
            for p, sel in parents_with_selected.items():
                if len(sel) < 2:
                    continue
                sel_sorted = sorted(sel)
                all_pos = dpt_sibling_positions[p]
                for k in range(len(sel_sorted) - 1):
                    a, b = sel_sorted[k], sel_sorted[k + 1]
                    for q in all_pos:
                        if q <= a:
                            continue
                        if q >= b:
                            break
                        if q not in selected:
                            extra += 1
            return extra

        def _subtrees_rooted_at(node: int, budget: int) -> list[frozenset[int]]:
            # All connected subtrees rooted at `node` with size in [1, budget].
            # The DP processes children one at a time; each child either
            # contributes nothing or one of its rooted subtrees, so the
            # children's contributions are disjoint and there are no
            # duplicates.
            base: list[frozenset[int]] = [frozenset({node})]
            if budget <= 1:
                return base
            states: list[tuple[int, frozenset[int]]] = [(0, frozenset())]
            for child in children_of[node]:
                new_states: list[tuple[int, frozenset[int]]] = []
                for size_so_far, pos_so_far in states:
                    new_states.append((size_so_far, pos_so_far))  # skip
                    remaining: int = budget - 1 - size_so_far
                    if remaining <= 0:
                        continue
                    for sub in _subtrees_rooted_at(child, remaining):
                        new_size: int = size_so_far + len(sub)
                        if new_size <= budget - 1:
                            new_states.append((new_size, pos_so_far | sub))
                states = new_states
            for _size_extra, pos_extra in states:
                if pos_extra:
                    base.append(frozenset({node}) | pos_extra)
            return base

        for parent_pos in range(len(state.sequence)):
            by_size: dict[int, list[frozenset[int]]] = {}
            for st in _subtrees_rooted_at(parent_pos, max_rule_size):
                if len(st) >= 2:
                    by_size.setdefault(len(st), []).append(st)
            if not by_size:
                continue
            for rule_number, rule in enumerate(RULES):
                for st in by_size.get(rule.size, []):
                    indices: tuple[int, ...] = tuple(sorted(st))
                    _try_candidate(rule_number, rule, indices, parent_pos)

        # Sibling-pair exception. Rules flagged with
        # consecutive_siblings_ok may additionally fire on a pair of
        # adjacent live hyperedges (positions p, p+1) that share a
        # common dep-parent position — i.e., adjacent dep-siblings
        # rather than a parent-child subtree. Lets the ":/J/."
        # conjunction rule combine coordinated items that the spaCy
        # parse left as siblings instead of as conj parent-child.
        parents_of_pos: dict[int, set[int]] = {}
        for _par, _ch in children_of.items():
            for _c in _ch:
                parents_of_pos.setdefault(_c, set()).add(_par)
        for rule_number, rule in enumerate(RULES):
            if not rule.consecutive_siblings_ok or rule.size != 2:
                continue
            for p in range(len(state.sequence) - 1):
                pa, pb = p, p + 1
                if not (parents_of_pos.get(pa, set()) & parents_of_pos.get(pb, set())):
                    continue
                _try_candidate(rule_number, rule, (pa, pb), pa)

        # Sliding-window per-position fallback for known-stranded atoms.
        # When a previous pass over the same sentence detected that some
        # atom was left bare at the top of the final hyperedge (or
        # force-joined by the iteration-level :/J/. fallback), that atom
        # is recorded in self._stranded_atoms. Here we generate extra
        # candidates for any position whose entity carries one of those
        # stranded atoms and whose position is otherwise uncovered by
        # the DPT-driven (or sibling-exception) candidates. For each
        # such position P we slide a window of size rule.size across P
        # for every rule whose first_type matches state.sequence[P].mtype();
        # apply_rule_indices scans pivot positions internally, so any
        # window where P validly pivots is accepted. With an empty
        # self._stranded_atoms (the first pass) the whole block is a
        # no-op.
        seq_len: int = len(state.sequence)
        if seq_len >= 2 and self._stranded_atoms:
            covered: set[int] = set()
            for _ca in cand_actions:
                covered.update(_ca[2])
            for pos in range(seq_len):
                if pos in covered:
                    continue
                pos_orig_atoms: set[UniqueAtom] = set()
                for a in state.sequence[pos].all_atoms():
                    ua = self.orig_atom.get(a)
                    if ua is not None:
                        pos_orig_atoms.add(ua)
                if not (pos_orig_atoms & self._stranded_atoms):
                    continue
                pos_mtype: str = state.sequence[pos].mtype()
                for rule_number, rule in enumerate(RULES):
                    if rule.first_type != pos_mtype:
                        continue
                    size: int = rule.size
                    if size > seq_len:
                        continue
                    for k in range(size):
                        ws: int = pos - k
                        we: int = ws + size - 1
                        if ws < 0 or we >= seq_len:
                            continue
                        indices = tuple(range(ws, we + 1))
                        _try_candidate(rule_number, rule, indices, pos)

        # Drop A if there exists B != A with A's signature a strict subset
        # of B's AND A and B share the same no_dangling status. Only
        # candidates whose rule has can_dominate=True can act as the
        # dominator B. The no_dangling parity guard prevents a dangling B
        # from suppressing a non-dangling A (and vice-versa) just because
        # B happens to cover more atoms — those are qualitatively
        # different parses, not redundant variants. Connectivity is no
        # longer compared — every candidate is dep-connected by
        # construction.
        #
        # Mandatory-rule guard: a mandatory candidate A is only
        # dominable by another *mandatory* candidate B — a non-mandatory
        # B can't suppress a mandatory A even if its signature is a
        # superset, since the mandatory rule is supposed to short-circuit
        # the iteration if it ranks first, and silently dropping it
        # would change downstream behavior.
        #
        # Connector-as-arg dominance (second rule): if A = (x a_1 a_2 ...)
        # has an atomic connector x and B = (b_1 ... x ...) carries x as
        # a direct argument, and B.mtype() matches A's rule first-slot
        # type, then A's rule could still fire on B later — producing
        # ((b_1 ... x ...) a_1 a_2 ...), the same final shape but with x
        # wrapped inside B first. Letting A win consumes x prematurely
        # and loses the wrap, so drop A unless A is strictly better than
        # B in (badness, distortion) — A survives only as a fallback
        # when it's the better-quality parse on those axes.
        n: int = len(cand_actions)

        def _norm_atom_str(s: str) -> str:
            parts: list[str] = s.split("/")
            if len(parts) >= 2:
                role: list[str] = parts[1].split(".")
                if len(role) >= 2:
                    parts[1] = role[0]
                    return "/".join(parts)
            return s

        cand_conn_norm: list[str | None] = []
        cand_arg_norms: list[set[str]] = []
        for k in range(n):
            edge_k: Hyperedge = cand_actions[k][1]
            if edge_k.atom:
                cand_conn_norm.append(None)
                cand_arg_norms.append(set())
                continue
            conn_k: Hyperedge = edge_k[0]
            cand_conn_norm.append(
                _norm_atom_str(cast(Atom, conn_k).atom_str) if conn_k.atom else None
            )
            args_k: set[str] = set()
            for arg in edge_k[1:]:
                if arg.atom:
                    args_k.add(_norm_atom_str(cast(Atom, arg).atom_str))
            cand_arg_norms.append(args_k)

        keep: list[bool] = [True] * n
        for i in range(n):
            for j in range(n):
                if i == j or not cand_can_dominate[j]:
                    continue
                if cand_no_dangling[i] != cand_no_dangling[j]:
                    continue
                if cand_mandatory[i] and not cand_mandatory[j]:
                    continue
                if cand_signatures[i] < cand_signatures[j]:
                    keep[i] = False
                    break
                a_conn: str | None = cand_conn_norm[i]
                if a_conn is None or a_conn not in cand_arg_norms[j]:
                    continue
                a_rule: Rule = RULES[cand_records[i].rule_index]
                b_edge: Hyperedge = cand_actions[j][1]
                if b_edge.mtype() != a_rule.first_type:
                    continue
                a_bd: tuple[int, int] = (
                    cand_actions[i][3],
                    cand_actions[i][4],
                )
                b_bd: tuple[int, int] = (
                    cand_actions[j][3],
                    cand_actions[j][4],
                )
                if a_bd < b_bd:
                    continue
                keep[i] = False
                break
        base_iter.dominated = [cand_records[i] for i in range(n) if not keep[i]]
        cand_actions = [cand_actions[i] for i in range(n) if keep[i]]
        base_iter.candidates = [cand_records[i] for i in range(n) if keep[i]]
        cand_mandatory = [cand_mandatory[i] for i in range(n) if keep[i]]

        if not cand_actions:
            base_iter.fallback_used = True
            if len(state.sequence) > 0:
                # Record the atoms consumed by this force-join so the
                # outer parse-orchestration loop can mark them stranded
                # and re-run with the sliding-window fallback enabled.
                for fj_edge in state.sequence[:2]:
                    for fj_atom in fj_edge.all_atoms():
                        fj_ua = self.orig_atom.get(fj_atom)
                        if fj_ua is not None:
                            self._force_joined_atoms.add(fj_ua)
                fallback: Hyperedge = hedge([":/J/.", *state.sequence[:2]])
                new_sequence: list[Hyperedge] = (
                    [fallback] if fallback else []
                ) + state.sequence[2:]
            else:
                new_sequence = []
            return _ParseState(
                sequence=new_sequence,
                total_badness=state.total_badness + _BADNESS_CRASH_PENALTY,
                total_score=state.total_score,
                total_distortion=state.total_distortion,
                iterations=[*state.iterations, base_iter],
                failed=True,
                seq_index=state.seq_index,
            )

        # Pick the single first-place candidate by
        # (badness ASC, distortion ASC, score DESC). A mandatory rule
        # that ranks first short-circuits the same way every other
        # candidate would have — there's only one survivor either way.
        indexed: list[
            tuple[int, tuple[int, Hyperedge, tuple[int, ...], int, int, int]]
        ] = list(enumerate(cand_actions))
        indexed.sort(key=lambda ic: (ic[1][3], ic[1][4], -ic[1][0]))
        cand_idx, (score, new_edge, indices, bad, distortion, parent_pos) = indexed[0]
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
                    indices=list(c.indices) if c.indices is not None else None,
                )
                for i, c in enumerate(base_iter.candidates)
            ],
            dominated=list(base_iter.dominated),
            fallback_used=False,
        )
        drop: set[int] = set(indices) - {parent_pos}
        new_sequence = [
            new_edge if k == parent_pos else e
            for k, e in enumerate(state.sequence)
            if k not in drop
        ]
        return _ParseState(
            sequence=new_sequence,
            total_badness=state.total_badness + bad,
            total_score=state.total_score + score,
            total_distortion=state.total_distortion + distortion,
            iterations=[*state.iterations, iter_for_child],
            failed=state.failed,
            seq_index=state.seq_index,
        )

    def _reduce_sequence(
        self, atom_sequence: list[Atom], seq_index: int = 0
    ) -> _ParseState | None:
        # Greedy reduction: at each step pick the single best rule
        # application (lowest (badness, distortion); ties broken by
        # higher score), apply it, and continue until the sequence has
        # one element left. _expand_state's crash-fallback path produces
        # a `failed=True` state with a :/J/.-joined edge when no rule
        # applies, so the loop always makes progress until the sequence
        # is fully reduced.
        state: _ParseState = _ParseState(
            sequence=list(atom_sequence),
            total_badness=0,
            total_score=0,
            seq_index=seq_index,
        )
        while len(state.sequence) >= 2:
            nxt: _ParseState | None = self._expand_state(state)
            if nxt is None:
                return None
            state = nxt
        return state

    def _hill_climb_atomization(
        self,
        seed_seq: list[Atom],
        alts: list[_AtomAlt],
        atom_traces: list[AtomTrace] | None = None,
    ) -> tuple[_ParseState | None, set[int], list[SubstitutionRound]]:
        # Hill-climbing label search. Starts from the all-top-1 seed;
        # while (badness, distortion) > (0, 0) and there are still
        # uncertain tokens to try, in each round attempts a single
        # substitution at every remaining uncertain position, picks the
        # lexicographic-best strict improvement, locks it in, and
        # repeats. `alts` is already sorted ascending by top-1
        # probability, so iteration order doubles as the
        # lowest-confidence-first tiebreak when several swaps tie.
        #
        # When `self._cur_trace is not None` (i.e. REPL `report` is on)
        # and `atom_traces` is supplied, we record each trial's edge,
        # badness, distortion, score, and the winner of each round into
        # the returned `rounds` list. Otherwise `rounds` stays empty —
        # callers (e.g. unit tests) that don't need reporting pay no
        # extra cost.
        current_seq: list[Atom] = list(seed_seq)
        remaining: list[_AtomAlt] = list(alts)
        substituted: set[int] = set()
        rounds: list[SubstitutionRound] = []
        record_trials: bool = self._cur_trace is not None and atom_traces is not None
        # `locked_subs` tracks the cumulative {tok_idx: alt_label} that
        # represents the current seed at the start of each round.
        # Snapshotted onto every recorded trial so `/sub <N>` can replay.
        locked_subs: dict[int, str] = {}
        trial_counter: int = 0

        best_state: _ParseState | None = self._reduce_sequence(current_seq)
        if best_state is None:
            return None, substituted, rounds
        # Lexicographic cost: (badness ASC, distortion ASC, -score ASC),
        # which corresponds to "minimize badness, then minimize distortion,
        # then maximize score".
        best_cost: tuple[int, int, int] = (
            best_state.total_badness,
            best_state.total_distortion,
            -best_state.total_score,
        )
        self.debug_msg(
            f"Seed search cost: badness={best_cost[0]} "
            f"distortion={best_cost[1]} score={-best_cost[2]}"
        )

        while (best_cost[0], best_cost[1]) != (0, 0) and remaining:
            winning_slot: int = -1
            winning_alt: _AtomAlt | None = None
            winning_state: _ParseState | None = None
            winning_cost: tuple[int, int, int] = best_cost
            round_record: SubstitutionRound | None = None
            if record_trials:
                round_record = SubstitutionRound(
                    round_idx=len(rounds),
                    seed_badness=best_cost[0],
                    seed_distortion=best_cost[1],
                    seed_score=-best_cost[2],
                )
                # No-substitution baseline so the report shows the
                # "keep current state" option alongside every attempted
                # swap. Marked is_winner=True later iff no swap strictly
                # improves on best_cost.
                round_record.trials.append(
                    SubstitutionTrial(
                        tok_idx=-1,
                        token_text="(no substitution)",
                        label_from="—",
                        label_to="—",
                        badness=best_state.total_badness,
                        distortion=best_state.total_distortion,
                        score=best_state.total_score,
                        edge_repr=" ".join(str(e) for e in best_state.sequence),
                        number=trial_counter,
                        substitutions=dict(locked_subs),
                    )
                )
                trial_counter += 1

            for slot, alt in enumerate(remaining):
                trial_seq: list[Atom] = list(current_seq)
                trial_seq[alt.seq_pos] = alt.uatom
                trial_state = self._reduce_sequence(trial_seq)
                if trial_state is None:
                    continue
                cost = (
                    trial_state.total_badness,
                    trial_state.total_distortion,
                    -trial_state.total_score,
                )
                self.debug_msg(
                    f"  trial substitute tok_idx={alt.tok_idx} → {alt.label}: "
                    f"badness={cost[0]} distortion={cost[1]} score={-cost[2]}"
                )
                if round_record is not None and atom_traces is not None:
                    round_record.trials.append(
                        SubstitutionTrial(
                            tok_idx=alt.tok_idx,
                            token_text=atom_traces[alt.tok_idx].token_text,
                            label_from=atom_traces[alt.tok_idx].predicted_type,
                            label_to=alt.label,
                            badness=trial_state.total_badness,
                            distortion=trial_state.total_distortion,
                            score=trial_state.total_score,
                            edge_repr=" ".join(str(e) for e in trial_state.sequence),
                            number=trial_counter,
                            substitutions={**locked_subs, alt.tok_idx: alt.label},
                        )
                    )
                    trial_counter += 1
                # Strict <, so the first slot to set winning_* wins any
                # later tie. `remaining` is sorted by top-1 prob asc, so
                # ties resolve to the most-uncertain position.
                if cost < winning_cost:
                    winning_cost = cost
                    winning_slot = slot
                    winning_alt = alt
                    winning_state = trial_state

            if winning_alt is None or winning_state is None:
                if round_record is not None:
                    round_record.improved = False
                    # No swap beats the baseline → no-substitution wins.
                    if round_record.trials:
                        round_record.trials[0].is_winner = True
                    rounds.append(round_record)
                break
            current_seq[winning_alt.seq_pos] = winning_alt.uatom
            substituted.add(winning_alt.tok_idx)
            locked_subs[winning_alt.tok_idx] = winning_alt.label
            if round_record is not None:
                round_record.improved = True
                for trial in round_record.trials:
                    if trial.tok_idx == winning_alt.tok_idx:
                        trial.is_winner = True
                rounds.append(round_record)
            remaining.pop(winning_slot)
            best_state = winning_state
            best_cost = winning_cost
            self.debug_msg(
                f"Locked substitution tok_idx={winning_alt.tok_idx} → "
                f"{winning_alt.label} (badness={best_cost[0]} "
                f"distortion={best_cost[1]} score={-best_cost[2]})"
            )

        self.debug_msg(f"Final sequence: {best_state.sequence}")
        self.debug_msg(f"Total badness: {best_state.total_badness}")
        self.debug_msg(f"Total distortion: {best_state.total_distortion}")
        self.debug_msg(f"Substituted token indices: {sorted(substituted)}")
        return best_state, substituted, rounds

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
