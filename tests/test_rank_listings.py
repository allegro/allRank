import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from allrank.inference.inference_utils import __rank_listings
from allrank.models.model import make_model


class ListBackedDataset(Dataset):
    def __init__(self, collection):
        self.collection = collection

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, idx):
        return self.collection[idx]


def test_rerank_listings():
    np.random.seed(42)

    n_listings = 2
    n_docs_per_listing = 5
    n_dimensions = 10

    X = [np.random.rand(n_docs_per_listing, n_dimensions).astype(np.float32) for _ in range(n_listings)]
    y_true = [np.random.randint(0, 1, size=len(x)) for x in X]
    indices = [np.repeat(0, len(x)) for x in X]

    fc_model = {"sizes": [10], "input_norm": False, "activation": None, "dropout": None}
    post_model = {"d_output": 1}
    model = make_model(fc_model, None, post_model, n_dimensions)

    dataloader = DataLoader(ListBackedDataset(list(zip(X, y_true, indices))), batch_size=2)
    listings_X, listings_y = __rank_listings(dataloader, model)

    assert len(listings_X) == len(X)
    assert len(listings_y) == len(y_true)

    for x in listings_X:
        assert x.shape[0] == n_docs_per_listing
        assert x.shape[1] == n_dimensions
    for y in listings_y:
        assert y.shape[0] == n_docs_per_listing
