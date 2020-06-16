import math

import torch
from pytest import approx
from torch.nn import BCEWithLogitsLoss

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import rankNet, rankNet_weightByGTDiff, rankNet_weightByGTDiff_pow


def rankNet_wrap(y_pred, y_true):
    return rankNet(torch.tensor([y_pred]), torch.tensor([y_true])).item()


def rankNet_weighted_wrap(y_pred, y_true):
    return rankNet_weightByGTDiff(torch.tensor([y_pred]), torch.tensor([y_true])).item()


def rankNet_weighted_pow_wrap(y_pred, y_true):
    return rankNet_weightByGTDiff_pow(torch.tensor([y_pred]), torch.tensor([y_true])).item()


def bce_wrap(y_pred, y_true, weight=None):
    if weight:
        weight = torch.tensor(weight)
    return BCEWithLogitsLoss(weight=weight)(torch.tensor([y_pred]), torch.tensor([y_true])).item()


def test_ranknet_onepair():
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]

    result = rankNet_wrap(y_pred, y_true)
    expected = bce_wrap(y_pred[0] - y_pred[1], 1.0)

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))


def test_ranknet_onepair_minus():
    y_pred = [0.2, 0.5]
    y_true = [1.0, 0.0]

    result = rankNet_wrap(y_pred, y_true)
    expected = bce_wrap(y_pred[0] - y_pred[1], 1.0)

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))


def test_ranknet_two_pairs():
    y_pred = [0.5, 0.2, 0.1]
    y_true = [1.0, 0.0, 0.0]

    result = rankNet_wrap(y_pred, y_true)
    expected = bce_wrap([0.3, 0.4], [1.0, 1.0])

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))


def test_ranknet_pair_reversed():
    y_pred = [0.2, 0.5]
    y_true = [0.0, 1.0]

    result = rankNet_wrap(y_pred, y_true)
    expected = bce_wrap(0.3, 1.0)

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))


def test_ranknet_pair_multirelevancy():
    y_pred = [0.2, 0.5]
    y_true = [0.0, 2.0]

    result = rankNet_wrap(y_pred, y_true)
    expected = bce_wrap(0.3, 1.0)

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))


def test_ranknet_onepair_masked():
    y_pred = [0.5, 0.2, 0.66]
    y_true = [1.0, 0.0, PADDED_Y_VALUE]

    result = rankNet_wrap(y_pred, y_true)
    expected = bce_wrap(y_pred[0] - y_pred[1], 1.0)

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))


def test_ranknet_two_pairs_weighted():
    y_pred = [0.5, 0.2, 0.1]
    y_true = [2.0, 1.0, 0.0]

    result = rankNet_weighted_wrap(y_pred, y_true)
    expected = bce_wrap([0.3, 0.4, 0.1], [1.0, 1.0, 1.0], weight=[1.0, 2.0, 1.0])

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))


def test_ranknet_two_pairs_weighted_pow():
    y_pred = [0.5, 0.2, 0.1]
    y_true = [2.0, 1.0, 0.0]

    result = rankNet_weighted_pow_wrap(y_pred, y_true)
    expected = bce_wrap([0.3, 0.4, 0.1], [1.0, 1.0, 1.0], weight=[3.0, 4.0, 1.0])

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))
