import math

import numpy as np
import torch
from pytest import approx
from scipy.special import softmax

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS, binary_listNet


def binary_listNet_wrap(y_pred, y_true, eps=1e-10):
    return binary_listNet(torch.tensor([y_pred]), torch.tensor([y_true]), eps).item()


def test_binary_listnet_simple():
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]

    result = binary_listNet_wrap(y_pred, y_true, eps=0.0)
    expected = - np.sum(y_true * np.log(softmax(y_pred)))

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))


def test_binary_listnet_stable_for_very_small_prediction():
    y_pred = [0.5, -1e30]
    y_true = [1.0, 0.0]

    result = binary_listNet_wrap(y_pred, y_true)
    expected = - np.sum(y_true * np.log(softmax(y_pred) + DEFAULT_EPS))

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected, abs=1e-9))


def test_binary_listnet_ignores_padded_value():
    y_pred = [0.5, 0.2, 0.5]
    y_true = [1.0, 0.0, PADDED_Y_VALUE]

    result = binary_listNet_wrap(y_pred, y_true)
    expected = - np.sum(y_true[:2] * np.log(softmax(y_pred[:2]) + DEFAULT_EPS))

    assert not math.isnan(result) and not math.isinf(result)
    assert (result == approx(expected))
