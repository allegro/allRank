from typing import Tuple, List

import numpy as np
import torch

from allrank.click_models.base import ClickModel
from allrank.data.dataset_loading import PADDED_Y_VALUE


def click_on_listings(listings, click_model, include_empty) -> Tuple[List[torch.Tensor], List[List[int]]]:
    X, y = listings
    clicks = [MaskedRemainMasked(click_model).click(listing) for listing in zip(X, y)]
    X_with_clicks = [[X, listing_clicks] for X, listing_clicks in list(zip(X, clicks)) if
                     (np.sum(listing_clicks > 0) > 0 or include_empty)]
    X, clicks = map(list, zip(*X_with_clicks))
    return X, clicks  # type: ignore


class MaskedRemainMasked(ClickModel):

    def __init__(self, delegate_click_model: ClickModel):
        self.delegate_click_model = delegate_click_model

    def click(self, documents):
        clicks = self.delegate_click_model.click(documents)
        _, y = documents
        mask = y == PADDED_Y_VALUE
        clicks[mask] = PADDED_Y_VALUE
        return clicks
