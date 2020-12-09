import math

import numpy as np
import torch
from pytest import approx

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import (
    ordinal,
    with_ordinals)


def loss_wrap(y_pred, y_true):
    return ordinal(torch.tensor([y_pred]), torch.tensor([y_true]), n=2).item()


def xe(true, pred):
    return - true * math.log(pred) - (1 - true) * math.log(1 - pred)


def test_ds_transform():
    y_true = np.array([2.0, 1.0, 0.0])
    result = with_ordinals(torch.tensor([y_true], dtype=torch.float), 2).tolist()
    expected = [[[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]]
    assert result == expected


def test_ordinal_single_doc():
    y_pred = [[0.8, 0.6]]
    y_true = [1.0]

    result = loss_wrap(y_pred, y_true)
    expected = np.mean([xe(1, 0.8) + xe(0, 0.6)])

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))


def test_ordinal_simple():
    y_pred = [[0.8, 0.7], [0.4, 0.3], [0.2, 0.1]]
    y_true = [2.0, 1.0, 0.0]

    result = loss_wrap(y_pred, y_true)
    expected = np.mean([xe(1, 0.8) + xe(1, 0.7), xe(1, 0.4) + xe(0, 0.3), xe(0, 0.2) + xe(0, 0.1)])

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))


def test_ordinal_single_doc_padded():
    y_pred = [[0.8, 0.6], [0.2, 0.1]]
    y_true = [1.0, PADDED_Y_VALUE]

    result = loss_wrap(y_pred, y_true)
    expected = np.mean([xe(1, 0.8) + xe(0, 0.6)])

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))
