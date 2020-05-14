from typing import Iterable

import numpy as np
from sklearn.datasets import dump_svmlight_file

from allrank.data.dataset_loading import PADDED_Y_VALUE


def write_to_libsvm_without_masked(path: str, X: Iterable[np.ndarray], y: Iterable[np.ndarray]) -> None:
    """
    This function writes given X's and y's as a libsvm format.
    It supports padded documents - they are removed from the written dataset.
    Listings are identified by a 'qid' column within the file.

    :param path: a path to save libsvm file
    :param X: Iterable of lists of document vectors
    :param y: Iterable of lists of document relevancies
    """
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
