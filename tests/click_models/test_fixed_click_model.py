from allrank.click_models.base import FixedClickModel
from tests.click_models import click


def test_fixed_click_model_single():
    click_model = FixedClickModel([0])
    assert click(click_model, [], [1]) == [1]
    assert click(click_model, [], [1, 2]) == [1, 0]
    assert click(click_model, [], [1, 2, 3]) == [1, 0, 0]


def test_fixed_click_model_multiple():
    assert click(FixedClickModel([0, 1]), [], [1, 2, 3, 4]) == [1, 1, 0, 0]
    assert click(FixedClickModel([0, 1, 2]), [], [1, 2, 3, 4]) == [1, 1, 1, 0]
    assert click(FixedClickModel([0, 2, 3]), [], [1, 2, 3, 4]) == [1, 0, 1, 1]
