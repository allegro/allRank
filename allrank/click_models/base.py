import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

import numpy as np
import torch


class ClickModel(ABC):
    """
    Base class for all click models. Specifies the click model contract
    """

    @abstractmethod
    def click(self, documents: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        """
        Applies a click model and returns the mask for documents.

        :rtype: np.ndarray [ number_of_documents ] -> a mask of the same length as the documents -
        defining whether a document was clicked (1), not clicked (0) or is a padded element (-1)

        :param documents: Tuple of :
           torch.Tensor [ number_of_documents, dimensionality_of_latent_vector ], representing features of documents
           torch.Tensor [ number_of_documents ] representing relevancy of documents
        """
        pass


class RandomClickModel(ClickModel):
    """
    This ClickModel clicks a configured number of times on random documents
    """

    def __init__(self, n_clicks: int):
        """

        :param n_clicks: number of documents that will be clicked
        """
        self.n_clicks = n_clicks

    def click(self, documents: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        X, y = documents
        clicks = np.random.choice(range(len(y)), size=self.n_clicks, replace=False)
        mask = np.zeros(len(y), dtype=bool)
        mask[clicks] = 1
        return mask


class FixedClickModel(ClickModel):
    """
    This ClickModel clicks on documents at fixed positions
    """

    def __init__(self, click_positions: List[int]):
        """

        :param click_positions: list of indices of documents that will be clicked
        """
        self.click_positions = click_positions

    def click(self, documents: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        X, y = documents
        clicks = np.zeros(len(y), dtype=bool)
        clicks[self.click_positions] = 1
        return clicks


class MultipleClickModel(ClickModel):
    """
    This click model uses one of given click models with given probability
    """

    def __init__(self, inner_click_models: List[ClickModel], probabilities: List[float]):
        """

        :param inner_click_models: list of click models to choose from
        :param probabilities: list of probabilities - must be of the same length as list of click models and sum to 1.0
        """
        self.inner_click_models = inner_click_models
        assert math.isclose(np.sum(probabilities), 1.0, abs_tol=1e-5), \
            f"probabilities should sum to one, but got {probabilities} which sums to {np.sum(probabilities)}"
        self.probabilities = np.array(probabilities).cumsum()

    def click(self, documents: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        index = np.argmax(np.random.rand() < self.probabilities)
        result = self.inner_click_models[index].click(documents)  # type: ignore
        return result


class ConditionedClickModel(ClickModel):
    """
    This click model allows to combine multiple click models with a logical funciton
    """

    def __init__(self, inner_click_models: List[ClickModel], combiner: Callable):
        """

        :param inner_click_models: list of click models to combine
        :param combiner: a function applied to the result of clicks from click models - e.g. np.all or np.any
        """
        self.inner_click_models = inner_click_models
        self.combiner = combiner

    def click(self, documents: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        clicks_from_click_models = [click_model.click(documents) for click_model in self.inner_click_models]
        return self.combiner(clicks_from_click_models, 0)


class MaxClicksModel(ClickModel):
    """
    This click model takes other click model and limits the number of clicks to given value
    effectively keeping top `max_clicks` clicks
    """

    def __init__(self, inner_click_model: ClickModel, max_clicks: int):
        """

        :param inner_click_model: a click model to generate clicks
        :param max_clicks: number of clicks that should be preserved
        """
        self.inner_click_model = inner_click_model
        self.max_clicks = max_clicks

    def click(self, documents: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        underlying_clicks = self.inner_click_model.click(documents)
        if self.max_clicks is not None:
            max_clicks_mask = underlying_clicks.cumsum() <= self.max_clicks
            return underlying_clicks * max_clicks_mask
        return underlying_clicks


class OnlyRelevantClickModel(ClickModel):
    """
    This ClickModel clicks on a document when its relevancy is greater that or equal to a predefined value

    """

    def __init__(self, relevancy_threshold: float):
        """
        :param relevancy_threshold: a minimum value of relevancy of a document to be clicked (inclusive)
        """
        self.relevancy_threshold = relevancy_threshold

    def click(self, documents: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        X, y = documents
        return np.array(y) >= self.relevancy_threshold
