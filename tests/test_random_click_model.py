import numpy as np

from allrank.click_models.base import RandomClickModel


def test_random_click_model_single():
    click_model = RandomClickModel(1)
    np.random.seed(42)
    assert click_model.click(([], [1])) == [1]
    assert click_model.click(([], [1, 2])).tolist() == [0, 1]
    assert click_model.click(([], [1, 2, 3])).tolist() == [0, 1, 0]


def test_random_click_model_multiple():
    np.random.seed(42)
    assert RandomClickModel(2).click(([], [1, 2, 3, 4])).tolist() == [0, 1, 0, 1]
    assert RandomClickModel(3).click(([], [1, 2, 3, 4])).tolist() == [1, 1, 0, 1]
    assert RandomClickModel(4).click(([], [1, 2, 3, 4])).tolist() == [1, 1, 1, 1]
