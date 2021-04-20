import os
from argparse import ArgumentParser, Namespace

import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file


def parse_args() -> Namespace:
    # default params
    FEATURES_WITHOUT_LOGARITHM = [
        5, 6, 7, 8, 9, 15, 19, 57, 58, 62, 75, 79, 85, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 121, 122, 127, 129, 130]
    FEATURES_NEGATIVE = [110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 123, 124]

    parser = ArgumentParser("Normalize features script")

    parser.add_argument("--ds_path", help="location of the dataset", required=True, type=str)

    parser.add_argument("--features_without_logarithm",
                        help="indices of features which are to be normalized without being logarithmized", required=False,
                        type=int, nargs="*", default=FEATURES_WITHOUT_LOGARITHM)

    parser.add_argument("--features_negative",
                        help="indices of features which are to be normalized with logarithm but their values can be negative",
                        required=False, type=int, nargs="*" , default=FEATURES_NEGATIVE)

    return parser.parse_args()


args = parse_args()

x_train, y_train, query_ids_train = load_svmlight_file(os.path.join(args.ds_path, "train.txt"), query_id=True)
x_test, y_test, query_ids_test = load_svmlight_file(os.path.join(args.ds_path, "test.txt"), query_id=True)
x_vali, y_vali, query_ids_vali = load_svmlight_file(os.path.join(args.ds_path, "vali.txt"), query_id=True)

x_train_transposed = x_train.toarray().T
x_test_transposed = x_test.toarray().T
x_vali_transposed = x_vali.toarray().T

x_train_normalized = np.zeros(x_train_transposed.shape)
x_test_normalized = np.zeros(x_test_transposed.shape)
x_vali_normalized = np.zeros(x_vali_transposed.shape)

eps_log = 1e-2
eps = 1e-6

for i, feat in enumerate(x_train_transposed):
    feature_vector_train = feat
    feature_vector_test = x_test_transposed[i, ]
    feature_vector_vali = x_vali_transposed[i, ]

    if i in args.features_negative:
        feature_vector_train = (-1) * feature_vector_train
        feature_vector_test = (-1) * feature_vector_test
        feature_vector_vali = (-1) * feature_vector_vali

    if i not in args.features_without_logarithm:
        # log only if all values >= 0
        if np.all(feature_vector_train >= 0) & np.all(feature_vector_test >= 0) & np.all(feature_vector_vali >= 0):
            feature_vector_train = np.log(feature_vector_train + eps_log)
            feature_vector_test = np.log(feature_vector_test + eps_log)
            feature_vector_vali = np.log(feature_vector_vali + eps_log)
        else:
            print("Some values of feature no. {} are still < 0 which is why the feature won't be normalized".format(i))

    mean = np.mean(feature_vector_train)
    std = np.std(feature_vector_train)
    feature_vector_train = (feature_vector_train - mean) / (std + eps)
    feature_vector_test = (feature_vector_test - mean) / (std + eps)
    feature_vector_vali = (feature_vector_vali - mean) / (std + eps)
    x_train_normalized[i, ] = feature_vector_train
    x_test_normalized[i, ] = feature_vector_test
    x_vali_normalized[i, ] = feature_vector_vali

ds_normalized_path = "{}_normalized".format(args.ds_path)
os.makedirs(ds_normalized_path, exist_ok=True)

train_normalized_path = os.path.join(ds_normalized_path, "train.txt")
with open(train_normalized_path, "w"):
    dump_svmlight_file(x_train_normalized.T, y_train, train_normalized_path, query_id=query_ids_train)

test_normalized_path = os.path.join(ds_normalized_path, "test.txt")
with open(test_normalized_path, "w"):
    dump_svmlight_file(x_test_normalized.T, y_test, test_normalized_path, query_id=query_ids_test)

vali_normalized_path = os.path.join(ds_normalized_path, "vali.txt")
with open(vali_normalized_path, "w"):
    dump_svmlight_file(x_vali_normalized.T, y_vali, vali_normalized_path, query_id=query_ids_vali)

print("Dataset with normalized features saved here: {}.".format(ds_normalized_path))
