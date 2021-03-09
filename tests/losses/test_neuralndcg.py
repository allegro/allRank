import math

from functools import partial
from pytest import approx

from allrank.data.dataset_loading import PADDED_Y_VALUE
from tests.losses.utils import neuralNDCG_wrap, ndcg_wrap


test_cases = [{"stochastic": False, "transposed": False},
              {"stochastic": True, "transposed": False},
              {"stochastic": False, "transposed": True},
              {"stochastic": True, "transposed": True}]


def test_neuralNDCG_simple():
    for tc in test_cases:
        neuralNDCG_simple(partial(neuralNDCG_wrap, **tc))


def neuralNDCG_simple(fun):
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]

    result = fun(y_pred, y_true)
    expected = ndcg_wrap(y_pred, y_true)

    assert math.isfinite(result)
    assert (-1 * result == approx(expected))


def test_neuralNDCG_longer():
    for tc in test_cases:
        neuralNDCG_longer(partial(neuralNDCG_wrap, **tc))


def neuralNDCG_longer(fun):
    y_pred = [0.5, 0.2, 0.1, 0.4, 1.0, -1.0, 0.63]
    y_true = [1.0, 2.0, 2.0, 4.0, 1.0, 4.0, 3.0]

    result = fun(y_pred, y_true)
    expected = ndcg_wrap(y_pred, y_true)

    assert math.isfinite(result)
    assert (-1 * result == approx(expected))


def test_neuralNDCG_stable_for_very_small_prediction():
    for tc in test_cases:
        neuralNDCG_stable_for_very_small_prediction(partial(neuralNDCG_wrap, **tc))


def neuralNDCG_stable_for_very_small_prediction(fun):
    y_pred = [0.5, -1e30]
    y_true = [1.0, 0.0]

    result = fun(y_pred, y_true)
    expected = ndcg_wrap(y_pred, y_true)

    assert math.isfinite(result)
    assert (-1 * result == approx(expected))


def test_neuralNDCG_ignores_padded_value():
    for tc in test_cases:
        neuralNDCG_ignores_padded_value(partial(neuralNDCG_wrap, **tc))


def neuralNDCG_ignores_padded_value(fun):
    y_pred = [0.5, 0.2, 0.1, 0.4, 1.0, -1.0, 0.63, 1., 0.5, 0.3]
    y_true = [1.0, 2.0, 2.0, 4.0, 1.0, 4.0, 3.0, PADDED_Y_VALUE, PADDED_Y_VALUE, PADDED_Y_VALUE]

    result = fun(y_pred, y_true, temperature=0.001)
    expected = ndcg_wrap(y_pred, y_true)

    assert math.isfinite(result)
    assert (-1 * result == approx(expected))


def test_neuralNDCG_at_3():
    for tc in test_cases:
        neuralNDCG_at_3(partial(neuralNDCG_wrap, **tc))


def neuralNDCG_at_3(fun):
    y_pred = [0.5, 0.2, 0.1, 0.4, 1.0, -1.0, 0.63]
    y_true = [1.0, 2.0, 2.0, 4.0, 1.0, 4.0, 3.0]
    ats = 3

    result = fun(y_pred, y_true, k=ats)
    expected = ndcg_wrap(y_pred, y_true, ats=[ats])

    assert math.isfinite(result)
    assert (-1 * result == approx(expected))
