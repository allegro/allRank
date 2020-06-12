import numpy as np

from allrank.click_models.base import OnlyRelevantClickModel
from tests.click_models import click


def test_only_relevant_click_model():
    click_model = OnlyRelevantClickModel(1)
    assert click(click_model, np.array([[0, 1]]), [1]) == [1]
    assert click(click_model, np.array([[0, 1]]), [0]) == [0]
    assert click(click_model, np.array([[1, 1], [1, 0], [0, 0]]), [1, 0, 0]) == [1, 0, 0]


def test_only_relevant_above_threshold_click_model():
    click_model = OnlyRelevantClickModel(2)
    assert click(click_model, np.array([[0, 1]]), [2]) == [1]
    assert click(click_model, np.array([[0, 1]]), [1]) == [0]
    assert click(click_model, np.array([[0, 1]]), [0]) == [0]
    assert click(click_model, np.array([[1, 1], [1, 0], [0, 0]]), [0, 1, 2]) == [0, 0, 1]
