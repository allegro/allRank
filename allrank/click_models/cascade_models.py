from typing import Tuple

import numpy as np
import torch
from scipy.spatial.distance import cdist

from allrank.click_models.base import ClickModel
from allrank.click_models.duplicate_aware import EverythingButDuplicatesClickModel
from allrank.data.dataset_loading import PADDED_Y_VALUE


class BaseCascadeModel(ClickModel):
    """
    This ClickModel simulates decaying probability of observing an item
    and clicks on an observed item given it's relevance is greater than or equal to a given threshold

    """

    def __init__(self, eta: float, threshold: float):
        """

        :param eta: the power to be applied over a result of a decay function (specified as 1/position)
                    to decide whether a document was observed
        :param threshold: a minimum value of relevancy of an observed document to be clicked (inclusive)
        """
        self.eta = eta
        self.threshold = threshold

    def click(self, documents: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        X, y = documents
        observed_mask = (1 / np.arange(1, len(y) + 1) ** self.eta) >= np.random.rand(len(y))
        return (y * observed_mask >= self.threshold).numpy()


class DiverseClicksModel(ClickModel):
    """
    A 'diverse-clicks' model from Seq2Slate paper https://arxiv.org/abs/1810.02019
    It clicks on documents from top to the bottom if:
      1. a delegate click model decides to click on the document (in the original paper - CascadeModel
      2. it is no closer than a defined percentile of distances to a previously clicked document
    """

    def __init__(self, inner_click_model, q_percentile=0.5):
        """

        :param inner_click_model: original, non-diversified click model
        :param q_percentile: a percentile of pairwise distances that will be used as a distance threshold to tell if a pair is a duplicate
        """
        self.inner_click_model = inner_click_model
        self.q_percentile = q_percentile

    def __pairwise_distances_list(self, X):
        dist = cdist(X, X, metric='euclidean')
        triu_indices = np.triu_indices(dist.shape[0] - 1)
        return dist[:-1, 1:][triu_indices]

    def click(self, documents: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        X, y = documents

        real_docs_mask = (y != PADDED_Y_VALUE)
        real_X = X[real_docs_mask, :]

        distances = self.__pairwise_distances_list(real_X)
        if len(distances) == 0:
            duplicate_margin = 0
        else:
            duplicate_margin = np.quantile(distances, q=self.q_percentile)

        def not_similar(x_vec, clicked_X):
            cX = clicked_X.copy()
            cX.append(x_vec)
            cX = torch.stack(cX, dim=0)
            cm = EverythingButDuplicatesClickModel(duplicate_margin)
            clicks = cm.click((cX, np.ones(len(cX))))
            last_element_clicked = clicks[-1]
            return last_element_clicked == 1

        relevant_for_click = self.inner_click_model.click(documents)

        clicked_Xs = []  # type: ignore
        indices_to_click = np.argwhere(relevant_for_click == 1)
        for idx_to_click in indices_to_click:
            idx_to_click = idx_to_click[0]
            X_to_click = X[idx_to_click]
            if not_similar(X_to_click, clicked_Xs):
                clicked_Xs.append(X_to_click)
            else:
                relevant_for_click[idx_to_click] = 0

        return relevant_for_click
