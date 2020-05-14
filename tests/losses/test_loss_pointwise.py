import math

import numpy as np
import torch
from pytest import approx

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import pointwise_rmse


def loss_wrap(y_pred, y_true, no_of_levels):
    return pointwise_rmse(torch.tensor([y_pred]), torch.tensor([y_true]), no_of_levels).item()


def test_pointwise_simple():
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]

    result = loss_wrap(y_pred, y_true, 1)
    expected = math.sqrt(np.mean([0.5 ** 2, 0.2 ** 2]))

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))


def test_pointwise_simple_padded():
    y_pred = [0.5, 0.2, 0.5]
    y_true = [1.0, 0.0, PADDED_Y_VALUE]

    result = loss_wrap(y_pred, y_true, 1)
    expected = math.sqrt(np.mean([0.5 ** 2, 0.2 ** 2]))

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))


def test_pointwise_multiple_levels():
    y_pred = [0.5, 0.2, 0.7, 0.8]
    y_true = [1.0, 0.0, 2.0, 3.0]

    result = loss_wrap(y_pred, y_true, 3)
    expected = math.sqrt(np.mean([0.5 ** 2, 0.6 ** 2, 0.1 ** 2, 0.6 ** 2]))

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))
