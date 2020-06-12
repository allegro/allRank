import numpy as np

from allrank.click_models.base import RandomClickModel
from tests.click_models import click


def test_random_click_model_single():
    click_model = RandomClickModel(1)
    np.random.seed(42)
    assert click(click_model, [], [1]) == [1]
    assert click(click_model, [], [1, 2]) == [0, 1]
    assert click(click_model, [], [1, 2, 3]) == [0, 1, 0]


def test_random_click_model_multiple():
    np.random.seed(42)
    assert click(RandomClickModel(2), [], [1, 2, 3, 4]) == [0, 1, 0, 1]
    assert click(RandomClickModel(3), [], [1, 2, 3, 4]) == [1, 1, 0, 1]
    assert click(RandomClickModel(4), [], [1, 2, 3, 4]) == [1, 1, 1, 1]
