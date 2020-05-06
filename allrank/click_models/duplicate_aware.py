import numpy as np
from allrank.click_models.base import ClickModel
from scipy.spatial.distance import cdist


class EverythingButDuplicatesClickModel(ClickModel):
    """
    this ClickModel clicks on every document, which was not previously clicked,
    if the distance between this document and any previous is larger than
    """

    def __init__(self, duplicate_margin=0, metric="euclidean"):
        self.duplicate_margin = duplicate_margin
        self.metric = metric

    def click(self, documents):
        X, y = documents
        dist = cdist(X, X, metric=self.metric)
        dist = np.triu(dist, k=1)
        np.fill_diagonal(dist, np.inf)
        indices = np.tril_indices(dist.shape[0])
        dist[indices] = np.inf
        return 1 * ((dist > self.duplicate_margin)).min(0)
