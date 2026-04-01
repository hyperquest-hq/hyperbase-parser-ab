import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from spacy.tokens import Span

from hyperbase_parser_ab.atomizer import Atomizer


class Alpha(object):
    def __init__(self, cases_str: str | None = None, use_atomizer: bool = False) -> None:
        if use_atomizer:
            self.atomizer: Atomizer | None = Atomizer()
        elif cases_str:
            self.atomizer = None

            X: list[tuple[str, str, str, str, str]] = []
            y: list[list[str]] = []

            for line in cases_str.strip().split('\n'):
                sline: str = line.strip()
                if len(sline) > 0:
                    row: list[str] = sline.strip().split('\t')
                    true_value: str = row[0]
                    tag: str = row[3]
                    dep: str = row[4]
                    hpos: str = row[6]
                    hdep: str = row[8]
                    pos_after: str = row[19]

                    y.append([true_value])
                    X.append((tag, dep, hpos, hdep, pos_after))

            if len(y) > 0:
                self.empty: bool = False

                self.encX: OneHotEncoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                self.encX.fit(np.array(X))
                self.ency: OneHotEncoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                self.ency.fit(np.array(y))

                X_: NDArray | spmatrix = self.encX.transform(np.array(X))
                y_: NDArray | spmatrix = self.ency.transform(np.array(y))

                self.clf: RandomForestClassifier = RandomForestClassifier(random_state=777)
                self.clf.fit(X_, y_)
        else:
            self.empty = True

    def predict(self, sentence: Span, features: list[tuple[str, str, str, str, str]]) -> tuple[str, ...] | list[str]:
        if self.atomizer:
            preds: list[tuple[str, str]] = self.atomizer.atomize(
                sentence=str(sentence),
                tokens=[str(token) for token in sentence])
            atom_types: list[str] = [pred[1] for pred in preds]

            # force known cases
            for i in range(len(atom_types)):
                if sentence[i].pos_ == 'VERB':
                    atom_types[i] = 'P'
            return atom_types
        else:
            # an empty classifier always predicts 'C'
            if self.empty:
                return tuple('C' for _ in range(len(features)))
            _features: NDArray | spmatrix = self.encX.transform(np.array(features))
            preds_arr: NDArray | spmatrix = self.ency.inverse_transform(self.clf.predict(_features))
            return tuple(pred[0] if pred else 'C' for pred in preds_arr)
