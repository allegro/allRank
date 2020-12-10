from typing import Tuple, Union

import numpy as np
import torch
from scipy.spatial.distance import cdist

from allrank.click_models.base import ClickModel


class EverythingButDuplicatesClickModel(ClickModel):
    """
    This ClickModel clicks on every document, which was not previously clicked,
    if the distance between this document and any previous is larger than given margin in given metric
    """

    def __init__(self, duplicate_margin: float = 0, metric: str = "euclidean"):
        """

        :param duplicate_margin: a margin to tell whether a pair of documents is treated as a duplicate.
            If the distance is less than or equal to this value - this marks a duplicate
        :param metric: a metric in which pairwise distances are calculated
            (metric must be supported by `scipy.spatial.distance.cdist`)
        """
        self.duplicate_margin = duplicate_margin
        self.metric = metric

    def click(self, documents: Tuple[torch.Tensor, Union[torch.Tensor, np.ndarray]]) -> np.ndarray:
        X, y = documents
        dist = cdist(X, X, metric=self.metric)
        dist = np.triu(dist, k=1)
        np.fill_diagonal(dist, np.inf)
        indices = np.tril_indices(dist.shape[0])
        dist[indices] = np.inf
        return 1 * (dist > self.duplicate_margin).min(0)
