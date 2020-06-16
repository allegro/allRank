import numpy as np

from allrank.click_models.cascade_models import BaseCascadeModel
from tests.click_models import click


def test_base_cascade_model_no_eta():
    click_model = BaseCascadeModel(0.0, 1)
    assert click(click_model, [], [1]) == [1]
    assert click(click_model, [], [1, 2]) == [1, 1]
    assert click(click_model, [], [1, 2, 3]) == [1, 1, 1]


def test_base_cascade_model_below_threshold():
    y = [1, 2, 0, 4, 3]
    assert click(BaseCascadeModel(0.0, 1), [], y) == [1, 1, 0, 1, 1]
    assert click(BaseCascadeModel(0.0, 2), [], y) == [0, 1, 0, 1, 1]
    assert click(BaseCascadeModel(0.0, 4), [], y) == [0, 0, 0, 1, 0]


def test_base_cascade_model_eta():
    np.random.seed(42)
    click_model_1 = BaseCascadeModel(0.3, 1)
    click_model_2 = BaseCascadeModel(0.5, 1)
    assert click(click_model_1, [], [1, 2]) == [1, 0]
    assert click(click_model_1, [], [1, 2, 3]) == [1, 1, 1]
    assert click(click_model_1, [], [1, 2, 3, 4]) == [1, 1, 0, 1]
    assert click(click_model_2, [], [1, 2]) == [1, 1]
    assert click(click_model_2, [], [1, 2, 3]) == [1, 0, 1]
    assert click(click_model_2, [], [1, 2, 3, 4]) == [1, 1, 1, 0]


def test_base_cascade_model_obs_irrelevant():
    np.random.seed(42)
    y = [1, 2, 0, 4, 3]
    assert click(BaseCascadeModel(0.3, 0), [], y) == [1, 1, 1, 1, 1]
    assert click(BaseCascadeModel(0.3, 1), [], y) == [1, 1, 0, 1, 0]
    assert click(BaseCascadeModel(0.3, 3), [], y) == [0, 0, 0, 1, 1]
    assert click(BaseCascadeModel(0.3, 4), [], y) == [0, 0, 0, 1, 0]
