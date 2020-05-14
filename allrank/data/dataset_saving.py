import numpy as np
from sklearn.datasets import dump_svmlight_file

from allrank.data.dataset_loading import PADDED_Y_VALUE


def write_to_libsvm_without_masked(path: str, X, y):
    Xs = []
    ys = []
    qids = []
    qid = 0
    for X, y in zip(X, y):
        mask = y != PADDED_Y_VALUE
        Xs.append(X[mask])
        ys.append(y[mask])
        qids.append(np.repeat(qid, len(y[mask])))
        qid += 1
    Xs = np.vstack(Xs)
    ys = np.concatenate(ys)
    qids = np.concatenate(qids)
    dump_svmlight_file(Xs, ys, path, query_id=qids)
