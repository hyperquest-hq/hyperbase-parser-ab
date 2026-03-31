import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from hyperbase_parser_ab.atomizer import Atomizer


class Alpha(object):
    def __init__(self, cases_str=None, use_atomizer=False):
        if use_atomizer:
            self.atomizer = Atomizer()
        elif cases_str:
            self.atomizer = None

            X = []
            y = []

            for line in cases_str.strip().split('\n'):
                sline = line.strip()
                if len(sline) > 0:
                    row = sline.strip().split('\t')
                    true_value = row[0]
                    tag = row[3]
                    dep = row[4]
                    hpos = row[6]
                    hdep = row[8]
                    pos_after = row[19]

                    y.append([true_value])
                    X.append((tag, dep, hpos, hdep, pos_after))

            if len(y) > 0:
                self.empty = False

                self.encX = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                self.encX.fit(np.array(X))
                self.ency = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                self.ency.fit(np.array(y))

                X_ = self.encX.transform(np.array(X))
                y_ = self.ency.transform(np.array(y))

                self.clf = RandomForestClassifier(random_state=777)
                self.clf.fit(X_, y_)
        else:
            self.empty = True

    def predict(self, sentence, features):
        if self.atomizer:
            preds = self.atomizer.atomize(
                sentence=str(sentence),
                tokens=[str(token) for token in sentence])
            atom_types = [pred[1] for pred in preds]

            # force known cases
            for i in range(len(atom_types)):
                if sentence[i].pos_ == 'VERB':
                    atom_types[i] = 'P'
            return atom_types
        else:
            # an empty classifier allways predicts 'C'
            if self.empty:
                return tuple('C' for _ in range(len(features)))
            _features = self.encX.transform(np.array(features))
            preds = self.ency.inverse_transform(self.clf.predict(_features))
            return tuple(pred[0] if pred else 'C' for pred in preds)
