from typing import Union, List

import numpy as np
import torch

from allrank.click_models.base import ClickModel


def click(click_model: ClickModel, X: Union[list, np.ndarray], y: Union[list, np.ndarray]) -> List[int]:
    clicks = click_model.click((torch.tensor(X), torch.tensor(y)))
    assert isinstance(clicks, np.ndarray)
    return clicks.tolist()
