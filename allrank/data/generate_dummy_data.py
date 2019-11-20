from argparse import ArgumentParser
from typing import Any, Dict
from typing import Tuple

import os
import numpy as np
from sklearn.datasets import dump_svmlight_file


def generate_dummy_data(
        num_queries: int, results_len: int, num_labels: int, num_features: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate dummy dataset to be dumped in libsvm format.
    """
    X = np.random.rand(num_queries * results_len, num_features)
    y = np.random.randint(0, num_labels-1, num_queries * results_len)
    qid = np.repeat(np.arange(0, num_queries), results_len)
    return X, y, qid


def parse_args() -> Dict[str, Any]:
    parser = ArgumentParser("Dummy data")
    parser.add_argument("--num_queries", help="Number of queries.", default=6)
    parser.add_argument("--results_len", help="Length of results list for a single query.", default=20)
    parser.add_argument("--num_labels", help="Number of relevance levels.", default=5)
    parser.add_argument("--num_features", help="Number of features of a single item", default=20)
    return vars(parser.parse_args())


if __name__ == '__main__':

    args = parse_args()
    X_train, y_train, qid_train = generate_dummy_data(**args)
    X_val, y_val, qid_val = generate_dummy_data(**args)

    os.makedirs("dummy_data", exist_ok=True)
    dump_svmlight_file(X_train, y_train, os.path.join("dummy_data", "train.txt"), query_id=qid_train)
    dump_svmlight_file(X_val, y_val, os.path.join("dummy_data", "vali.txt"), query_id=qid_val)
