import numpy as np

from allrank.click_models.base import FixedClickModel, MultipleClickModel, ConditionedClickModel


def test_click_model_should_use_all_click_models():
    np.random.seed(42)

    click_model_0 = FixedClickModel([0])
    click_model_1 = FixedClickModel([1])
    click_model = MultipleClickModel([click_model_0, click_model_1], probabilities=[0.5, 0.5])
    clicks = np.array([
        click_model.click(([], [1, 2])).tolist()
        for _ in range(20000)
    ])
    assert 9950 < np.sum(clicks[:, 0]) < 10050
    assert 9950 < np.sum(clicks[:, 1]) < 10050


def test_click_model_should_combine_click_models_and():
    click_model_0 = FixedClickModel([0, 1])
    click_model_1 = FixedClickModel([1, 2])
    click_model = ConditionedClickModel([click_model_0, click_model_1], np.all)
    clicks = click_model.click(([], [1, 2, 3])).tolist()
    assert clicks == [0, 1, 0]


def test_click_model_should_combine_click_models_or():
    click_model_0 = FixedClickModel([0, 1])
    click_model_1 = FixedClickModel([1, 2])
    click_model = ConditionedClickModel([click_model_0, click_model_1], np.any)
    clicks = click_model.click(([], [1, 2, 3, 4])).tolist()
    assert clicks == [1, 1, 1, 0]
