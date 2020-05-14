from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class ClickModel(ABC):

    @abstractmethod
    def click(self, documents: Tuple[np.ndarray, np.ndarray]):
        """
        return the mask for documents 1 if clicked, else 0.

        :rtype: np.ndarray [ number_of_clicks ] -> a 0/1/-1 mask of length same as documents -
        representing whether document was clicked (1) or not (0) or remained masked (-1)

        :param documents: Tuple of :
           np.ndarray [ number_of_documents, dimensionality_of_latent_vector ], representing features of documents
           np.ndarray [ number_of_documents ] representing relevancy of documents
        """
        pass


class RandomClickModel(ClickModel):
    """
    this ClickModel clicks on a random list of documents with a fixed number specified as a param

    """

    def __init__(self, n_clicks):
        self.n_clicks = n_clicks

    def click(self, documents):
        X, y = documents
        clicks = np.random.choice(range(len(y)), size=self.n_clicks, replace=False)
        mask = np.repeat(0, len(y))
        mask[clicks] = 1
        return mask


class FixedClickModel(ClickModel):
    """
    this ClickModel clicks on a fixed list of documents specified as a param

    """

    def __init__(self, click_positions):
        self.click_positions = click_positions

    def click(self, documents):
        X, y = documents
        clicks = np.repeat(0, len(y))
        clicks[self.click_positions] = 1
        return clicks


class MultipleClickModel(ClickModel):

    def __init__(self, click_models, probabilities):
        self.click_models = click_models
        assert np.sum(probabilities) >= 0.99999 and np.sum(
            probabilities) <= 1.000001, "probabilities should sum to one, but got {} which sums to {}".format(
            probabilities, np.sum(probabilities)
        )
        self.probabilities = np.array(probabilities).cumsum()

    def click(self, documents):
        index = np.argmax(np.random.rand() < self.probabilities)
        result = self.click_models[index].click(documents)
        # print("finally model from index {} got {} clicks {}".format(index, np.sum(result, -1), result))
        return result


class ConditionedClickModel(ClickModel):
    """
        This click model allows to combine multiple click models with a logical funciton

        :param click_models: list of click models to combine
        :param combiner: a function applied to the result of clicks from click models - e.g. np.all or np.any
        """

    def __init__(self, click_models, combiner):
        self.click_models = click_models
        self.combiner = combiner

    def click(self, documents):
        clicks_from_click_models = [click_model.click(documents) for click_model in self.click_models]
        return self.combiner(clicks_from_click_models, 0)


def combine_with_and(*click_models: ClickModel):
    return ConditionedClickModel(click_models, np.all)


def combine_with_or(click_models: List[ClickModel]):
    return ConditionedClickModel(click_models, np.any)


class MaxClicksModel(ClickModel):

    def __init__(self, click_model, max_clicks):
        self.click_model = click_model
        self.max_clicks = max_clicks

    def click(self, documents):
        underlying_clicks = self.click_model.click(documents)
        if self.max_clicks is not None:
            max_clicks_mask = underlying_clicks.cumsum() <= self.max_clicks
            return underlying_clicks * max_clicks_mask
        return underlying_clicks


class OnlyRelevantClickModel(ClickModel):
    """
    this ClickModel clicks on a document:
    1. when its relevancy is at least of predefined value

    """

    def __init__(self, relevancy_threshold=1):
        self.relevancy_threshold = relevancy_threshold

    def click(self, documents):
        X, y = documents
        return np.array(y) >= self.relevancy_threshold
