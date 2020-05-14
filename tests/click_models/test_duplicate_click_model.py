import numpy as np

from allrank.click_models.duplicate_aware import EverythingButDuplicatesClickModel


def test_feature_click_model_everything():
    click_model = EverythingButDuplicatesClickModel()
    assert click_model.click((np.array([[0, 1]]), [])).tolist() == [1]
    assert click_model.click((np.array([[1, 1], [1, 0]]), [])).tolist() == [1, 1]
    assert click_model.click((np.array([[1, 1], [1, 0], [0, 0]]), [])).tolist() == [1, 1, 1]


def test_feature_click_model_except_exact_duplicates():
    click_model = EverythingButDuplicatesClickModel()
    assert click_model.click((np.array([[0, 1]]), [])).tolist() == [1]
    assert click_model.click((np.array([[1, 1], [1, 1]]), [])).tolist() == [1, 0]
    assert click_model.click((np.array([[1, 1], [1, 1], [0, 0]]), [])).tolist() == [1, 0, 1]


def test_feature_click_model_except_near_duplicates():
    click_model = EverythingButDuplicatesClickModel(0.1)
    assert click_model.click((np.array([[0, 1]]), [])).tolist() == [1]
    assert click_model.click((np.array([[1, 1], [1, 1]]), [])).tolist() == [1, 0]
    assert click_model.click((np.array([[1, 1], [1, 0.99], [1, 0.8]]), [])).tolist() == [1, 0, 1]
