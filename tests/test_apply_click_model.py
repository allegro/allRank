import numpy as np
from allrank.click_models.base import RandomClickModel
from allrank.click_models.click_utils import click_on_listings


def test_click_on_listings():
    np.random.seed(42)

    n_listings = 5
    n_docs_per_listing = 5
    n_dimensions = 10

    X = np.random.rand(n_listings, n_docs_per_listing, n_dimensions).astype(np.float32)
    y = [np.random.randint(0, 4, size=len(x)) for x in X]

    n_clicks = 2
    click_model = RandomClickModel(n_clicks)

    listings_X, listings_y = click_on_listings((X, y), click_model, True)

    assert listings_X.shape == X.shape  # checks that X has shape
    assert (listings_X == X).all()
    assert len(listings_y) == len(y)
    assert (np.sum(listings_y, axis=1) == np.repeat(n_clicks, n_listings)).all()


def test_click_on_listings_without_empty():
    np.random.seed(42)

    X = np.array([[[-1.0]], [[1.0]]])
    y = [np.array([0]), np.array([0])]

    click_model = RandomClickModel(2)

    listings_X, listings_y = click_on_listings((X, y), click_model, include_empty=False)

    assert listings_X.shape == X[1:].shape  # checks that X has shape
    assert (listings_X == X[1:]).all()
    assert listings_y == [[1]]
