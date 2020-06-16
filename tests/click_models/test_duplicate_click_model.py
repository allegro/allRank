import numpy as np

from allrank.click_models.duplicate_aware import EverythingButDuplicatesClickModel
from tests.click_models import click


def test_feature_click_model_everything():
    click_model = EverythingButDuplicatesClickModel()
    assert click(click_model, np.array([[0, 1]]), []) == [1]
    assert click(click_model, np.array([[1, 1], [1, 0]]), []) == [1, 1]
    assert click(click_model, np.array([[1, 1], [1, 0], [0, 0]]), []) == [1, 1, 1]


def test_feature_click_model_except_exact_duplicates():
    click_model = EverythingButDuplicatesClickModel()
    assert click(click_model, np.array([[0, 1]]), []) == [1]
    assert click(click_model, np.array([[1, 1], [1, 1]]), []) == [1, 0]
    assert click(click_model, np.array([[1, 1], [1, 1], [0, 0]]), []) == [1, 0, 1]


def test_feature_click_model_except_near_duplicates():
    click_model = EverythingButDuplicatesClickModel(0.1)
    assert click(click_model, np.array([[0, 1]]), []) == [1]
    assert click(click_model, np.array([[1, 1], [1, 1]]), []) == [1, 0]
    assert click(click_model, np.array([[1, 1], [1, 0.99], [1, 0.8]]), []) == [1, 0, 1]
