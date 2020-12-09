import numpy as np

from allrank.click_models.base import RandomClickModel, OnlyRelevantClickModel
from allrank.click_models.click_utils import click_on_slates


def test_click_on_slates():
    np.random.seed(42)

    n_slates = 5
    n_docs_per_slate = 5
    n_dimensions = 10

    X = np.random.rand(n_slates, n_docs_per_slate, n_dimensions).astype(np.float32)
    y = np.vstack([np.random.randint(0, 4, size=len(x)) for x in X])

    n_clicks = 2
    click_model = RandomClickModel(n_clicks)

    slates_X, slates_y = click_on_slates((X, y), click_model, True)

    assert len(slates_X) == X.shape[0]
    assert (slates_X == X).all()
    assert len(slates_y) == len(y)
    assert (np.sum(slates_y, axis=1) == np.repeat(n_clicks, n_slates)).all()


def test_click_on_slates_without_empty():
    np.random.seed(42)

    X = np.array([[[-1.0]], [[1.0]]])
    y = np.vstack([np.array([0]), np.array([1])])

    click_model = OnlyRelevantClickModel(1)

    slates_X, slates_y = click_on_slates((X, y), click_model, include_empty=False)

    assert len(slates_X) == X[1:].shape[0]
    assert (slates_X == X[1:]).all()
    assert slates_y == [[1]]
