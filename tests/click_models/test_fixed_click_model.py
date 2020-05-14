from allrank.click_models.base import FixedClickModel


def test_fixed_click_model_single():
    click_model = FixedClickModel([0])
    assert click_model.click(([], [1])).tolist() == [1]
    assert click_model.click(([], [1, 2])).tolist() == [1, 0]
    assert click_model.click(([], [1, 2, 3])).tolist() == [1, 0, 0]


def test_fixed_click_model_multiple():
    assert FixedClickModel([0, 1]).click(([], [1, 2, 3, 4])).tolist() == [1, 1, 0, 0]
    assert FixedClickModel([0, 1, 2]).click(([], [1, 2, 3, 4])).tolist() == [1, 1, 1, 0]
    assert FixedClickModel([0, 2, 3]).click(([], [1, 2, 3, 4])).tolist() == [1, 0, 1, 1]
