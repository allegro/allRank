import numpy as np

from allrank.click_models.cascade_models import BaseCascadeModel


def test_base_cascade_model_no_eta():
    click_model = BaseCascadeModel(0.0, 1)
    assert click_model.click(([], [1])).tolist() == [1]
    assert click_model.click(([], [1, 2])).tolist() == [1, 1]
    assert click_model.click(([], [1, 2, 3])).tolist() == [1, 1, 1]


def test_base_cascade_model_below_threshold():
    y = [1, 2, 0, 4, 3]
    assert BaseCascadeModel(0.0, 1).click(([], y)).tolist() == [1, 1, 0, 1, 1]
    assert BaseCascadeModel(0.0, 2).click(([], y)).tolist() == [0, 1, 0, 1, 1]
    assert BaseCascadeModel(0.0, 4).click(([], y)).tolist() == [0, 0, 0, 1, 0]


def test_base_cascade_model_eta():
    np.random.seed(42)
    click_model_1 = BaseCascadeModel(0.3, 1)
    click_model_2 = BaseCascadeModel(0.5, 1)
    assert click_model_1.click(([], [1, 2])).tolist() == [1, 0]
    assert click_model_1.click(([], [1, 2, 3])).tolist() == [1, 1, 1]
    assert click_model_1.click(([], [1, 2, 3, 4])).tolist() == [1, 1, 0, 1]
    assert click_model_2.click(([], [1, 2])).tolist() == [1, 1]
    assert click_model_2.click(([], [1, 2, 3])).tolist() == [1, 0, 1]
    assert click_model_2.click(([], [1, 2, 3, 4])).tolist() == [1, 1, 1, 0]


def test_base_cascade_model_obs_irrelevant():
    np.random.seed(42)
    y = [1, 2, 0, 4, 3]
    assert BaseCascadeModel(0.3, 0).click(([], y)).tolist() == [1, 1, 1, 1, 1]
    assert BaseCascadeModel(0.3, 1).click(([], y)).tolist() == [1, 1, 0, 1, 0]
    assert BaseCascadeModel(0.3, 3).click(([], y)).tolist() == [0, 0, 0, 1, 1]
    assert BaseCascadeModel(0.3, 4).click(([], y)).tolist() == [0, 0, 0, 1, 0]
