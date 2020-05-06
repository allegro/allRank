import numpy as np


def click_on_listings(listings, click_model, include_empty):
    X, y = listings
    clicks = [click_model.click(listing) for listing in zip(X, y)]
    X_with_clicks = list(zip(X, clicks))
    if not include_empty:
        X_with_clicks = [(X, listing_clicks) for X, listing_clicks in X_with_clicks if
                         (np.sum(listing_clicks) > 0 or include_empty)]
    X, clicks = list(zip(*X_with_clicks))
    return np.array(X), np.array(clicks)
