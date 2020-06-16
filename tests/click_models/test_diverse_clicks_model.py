import numpy as np

from allrank.click_models.cascade_models import BaseCascadeModel, DiverseClicksModel
from tests.click_models import click

base_click_model = BaseCascadeModel(0.0, 1)
click_model = DiverseClicksModel(base_click_model)


def test_diverse_clicks_model_simple():
    assert click(click_model, np.array([[0, 1]]), [1]) == [1]
    assert click(click_model, np.array([[0, 1], [0, 1]]), [1, 1]) == [1, 0]
    assert click(click_model, np.array([[0, 1], [0, 1], [1, 1]]), [1, 1, 1]) == [1, 0, 0]
    assert click(click_model, np.array([[0, 1], [0, 1], [2, 2], [1, 1]]), [1, 1, 1, 1]) == [1, 0, 1, 0]
