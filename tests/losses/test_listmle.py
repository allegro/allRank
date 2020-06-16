import math

import torch
from pytest import approx

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import listMLE


def listmle_wrap(y_pred, y_true):
    return listMLE(torch.tensor([y_pred]), torch.tensor([y_true])).item()


def test_listmle_ignores_padded_value():
    y_pred = [0.5, 0.3, 0.5]
    y_true = [1.0, 0.0, PADDED_Y_VALUE]

    result = listmle_wrap(y_pred, y_true)
    expected = 0.5981389284133911

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))
