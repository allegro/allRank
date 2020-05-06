import numpy as np
import torch
from allrank.inference.inference_utils import rank_listings
from allrank.models.model import make_model


def test_rerank_listings():
    np.random.seed(42)

    n_listings = 2
    n_docs_per_listing = 5
    n_dimensions = 10

    X = [np.random.rand(n_docs_per_listing, n_dimensions).astype(np.float32) for _ in range(n_listings)]
    y = [np.random.randint(0, 1, size=len(x)) for x in X]

    fc_model = {"sizes": [10], "input_norm": False, "activation": None, "dropout": None}
    post_model = {"d_output": 1}
    model = make_model(fc_model, None, post_model, n_dimensions)

    listings_X, listings_y = rank_listings((X, y), model, torch.device("cpu"))

    assert len(listings_X) == len(X)
    assert len(listings_y) == len(y)

    for x in listings_X:
        assert x.shape[0] == n_docs_per_listing
        assert x.shape[1] == n_dimensions
    # for y in listings_y:
    #     assert y.shape[0] == n_docs_per_listing
