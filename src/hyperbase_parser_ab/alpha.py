import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from spacy.tokens import Span

from hyperbase_parser_ab.atomizer import Atomizer


class Alpha:
    def __init__(
        self,
        cases_str: str | None = None,
        use_atomizer: bool = False,
        use_atomizer_subtype: bool = True,
        atomizer_model_path: str | None = None,
    ) -> None:
        self.use_atomizer_subtype: bool = use_atomizer_subtype
        if use_atomizer:
            self.atomizer: Atomizer | None = Atomizer(model_path=atomizer_model_path)
        elif cases_str:
            self.atomizer = None

            x: list[tuple[str, str, str, str, str]] = []
            y: list[list[str]] = []

            for line in cases_str.strip().split("\n"):
                sline: str = line.strip()
                if len(sline) > 0:
                    row: list[str] = sline.strip().split("\t")
                    true_value: str = row[0]
                    tag: str = row[3]
                    dep: str = row[4]
                    hpos: str = row[6]
                    hdep: str = row[8]
                    pos_after: str = row[19]

                    y.append([true_value])
                    x.append((tag, dep, hpos, hdep, pos_after))

            if len(y) > 0:
                self.empty: bool = False

                self.encX: OneHotEncoder = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                )
                self.encX.fit(np.array(x))
                self.ency: OneHotEncoder = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                )
                self.ency.fit(np.array(y))

                x_: NDArray | spmatrix = self.encX.transform(np.array(x))
                y_: NDArray | spmatrix = self.ency.transform(np.array(y))

                self.clf: RandomForestClassifier = RandomForestClassifier(
                    random_state=777
                )
                self.clf.fit(x_, y_)
        else:
            self.empty = True

    def predict(
        self, sentence: Span, features: list[tuple[str, str, str, str, str]]
    ) -> tuple[
        tuple[str, ...] | list[str],
        list[list[tuple[str, float]]],
    ]:
        if self.atomizer:
            preds = self.atomizer.atomize(
                sentence=str(sentence),
                tokens=[str(token) for token in sentence],
                top_k=3,
            )
            top_candidates: list[list[tuple[str, float]]] = [
                pred[1]
                for pred in preds  # type: ignore[misc]
            ]
            atom_types: list[str] = [cands[0][0] for cands in top_candidates]

            if not self.use_atomizer_subtype:
                # force known cases
                for i in range(len(atom_types)):
                    if sentence[i].pos_ == "VERB":
                        atom_types[i] = "P"
            return atom_types, top_candidates
        else:
            # an empty classifier always predicts 'C'
            if self.empty:
                return (
                    tuple("C" for _ in range(len(features))),
                    [[] for _ in features],
                )
            _features: NDArray | spmatrix = self.encX.transform(np.array(features))
            preds_arr: NDArray | spmatrix = self.ency.inverse_transform(
                self.clf.predict(_features)
            )
            return (
                tuple(pred[0] if pred else "C" for pred in preds_arr),
                [[] for _ in features],
            )

    def predict_batch(
        self,
        sentences: list[Span],
        features_list: list[list[tuple[str, str, str, str, str]]],
    ) -> list[tuple[tuple[str, ...] | list[str], list[list[tuple[str, float]]]]]:
        """Predict atom types for several sentences at once.

        With the atomizer, runs a single batched transformer forward
        pass over all input spans. Without it, falls back to per-item
        :meth:`predict` (the classifier path is cheap enough that
        batching it is not a priority)."""
        if not sentences:
            return []
        if self.atomizer:
            sentences_str: list[str] = [str(s) for s in sentences]
            tokens_list: list[list[str]] = [
                [str(token) for token in s] for s in sentences
            ]
            all_preds = self.atomizer.atomize_batch(
                sentences=sentences_str,
                tokens_list=tokens_list,
                top_k=3,
            )
            results: list[
                tuple[tuple[str, ...] | list[str], list[list[tuple[str, float]]]]
            ] = []
            for sent_span, preds in zip(sentences, all_preds, strict=True):
                top_candidates: list[list[tuple[str, float]]] = [
                    pred[1]
                    for pred in preds  # type: ignore[misc]
                ]
                atom_types: list[str] = [cands[0][0] for cands in top_candidates]
                if not self.use_atomizer_subtype:
                    for i in range(len(atom_types)):
                        if sent_span[i].pos_ == "VERB":
                            atom_types[i] = "P"
                results.append((atom_types, top_candidates))
            return results
        return [
            self.predict(sent, feats)
            for sent, feats in zip(sentences, features_list, strict=True)
        ]
