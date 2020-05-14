from typing import Tuple

import numpy as np
from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.click_models.base import ClickModel
from allrank.click_models.duplicate_aware import EverythingButDuplicatesClickModel
from scipy.spatial.distance import cdist


class BaseCascadeModel(ClickModel):
    """
    this ClickModel simulates decaying probability of observing an item
    and generates click when item's relevance is at least equal to a threshold

    """

    def __init__(self, eta: float, threshold: float):
        """

        :param eta: the power to be applied over a result of a decay function (specified as 1/position)
                    to decide whether a document was observed
        :param threshold: a minimum value of relevancy of an observed document to be clicked (inclusive)
        """
        self.eta = eta
        self.threshold = threshold

    def click(self, documents: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        X, y = documents
        observed_mask = (1 / np.arange(1, len(y) + 1) ** self.eta) >= np.random.rand(len(y))
        return (y * observed_mask >= self.threshold).astype(int)


class DiverseClicksModel(ClickModel):
    """
    A 'diverse-clicks' model from Seq2Slate paper https://arxiv.org/abs/1810.02019
    It clicks on a documents from top to the bottom if:
      1. a delegate click model decides to click on the document (in the original paper - CascadeModel
      2. it is not closer than defined percentile of distances to a previously clicked document
    """

    def __init__(self, base_click_model, q_percentile=0.5):
        """

        :param base_click_model: a base click model
        :param q_percentile: a percentile of pairwise distances that will be used as a distance threshold to tell if pair is a duplicate
        """
        self.base_click_model = base_click_model
        self.q_percentile = q_percentile

    def __pairwise_distances_list(self, X):
        dist = cdist(X, X, metric='euclidean')
        triu_indices = np.triu_indices(dist.shape[0] - 1)
        return dist[:-1, 1:][triu_indices]

    def click(self, documents: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
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
            cm = EverythingButDuplicatesClickModel(duplicate_margin)
            clicks = cm.click((cX, np.ones(len(cX))))
            last_element_clicked = clicks[-1]
            return last_element_clicked == 1

        relevant_for_click = self.base_click_model.click(documents)

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
