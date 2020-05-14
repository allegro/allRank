import numpy as np

from allrank.click_models.base import OnlyRelevantClickModel


def test_only_relevant_click_model():
    click_model = OnlyRelevantClickModel()
    assert click_model.click((np.array([[0, 1]]), [1])).tolist() == [1]
    assert click_model.click((np.array([[0, 1]]), [0])).tolist() == [0]
    assert click_model.click((np.array([[1, 1], [1, 0], [0, 0]]), [1, 0, 0])).tolist() == [1, 0, 0]


def test_only_relevant_above_threshold_click_model():
    click_model = OnlyRelevantClickModel(2)
    assert click_model.click((np.array([[0, 1]]), [2])).tolist() == [1]
    assert click_model.click((np.array([[0, 1]]), [1])).tolist() == [0]
    assert click_model.click((np.array([[0, 1]]), [0])).tolist() == [0]
    assert click_model.click((np.array([[1, 1], [1, 0], [0, 0]]), [0, 1, 2])).tolist() == [0, 0, 1]
