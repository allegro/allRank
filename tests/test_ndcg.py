import math

import torch
from pytest import approx

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.metrics import ndcg


def ndcg_wrap(y_pred, y_true, ats=None):
    return ndcg(torch.tensor([y_pred]), torch.tensor([y_true]), ats=ats).numpy()


def test_ndcg_simple_1():
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]

    result = ndcg_wrap(y_pred, y_true)

    assert (result == 1.0)


def test_ndcg_simple_2():
    y_pred = [0.5, 0.2]
    y_true = [0.0, 1.0]

    result = ndcg_wrap(y_pred, y_true)

    assert (result == 1 / math.log2(3))


def test_ndcg_zero_when_no_relevant():
    y_pred = [0.5, 0.2]
    y_true = [0.0, 0.0]

    result = ndcg_wrap(y_pred, y_true)

    assert (result == 0.0)


def test_ndcg_for_multiple_ats():
    y_pred = [0.5, 0.2, 0.1]
    y_true = [1.0, 0.0, 1.0]

    result = ndcg_wrap(y_pred, y_true, ats=[1, 2])

    ndcg_one_relevant_on_top = 1.0 / (1.0 + 1 / math.log2(3))
    expected = [1.0, ndcg_one_relevant_on_top]

    batch_0 = 0
    assert result[batch_0] == approx(expected)


def test_ndcg_with_padded_input():
    y_pred = [0.5, 0.2, 1.0]
    y_true = [1.0, 0.0, PADDED_Y_VALUE]

    result = ndcg_wrap(y_pred, y_true)

    assert result == 1.0


def test_ndcg_with_padded_input_2():
    y_pred = [0.5, 0.2, 1.0]
    y_true = [0.0, 1.0, PADDED_Y_VALUE]

    result = ndcg_wrap(y_pred, y_true)

    assert result == 1 / math.log2(3)
