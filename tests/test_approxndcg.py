import math

import torch
from pytest import approx

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses.approxNDCG import approxNDCGLoss


def test_approxndcg_ignores_padded():
    y_pred = [torch.tensor([[0.5, 0.3, 0.5]]), torch.tensor([[0.5, 0.3, 0.5, 1.0]])]
    y_true = [torch.tensor([[0.5, 0.3, 0.5]]), torch.tensor([[0.5, 0.3, 0.5, PADDED_Y_VALUE]])]

    result = approxNDCGLoss(y_pred[0], y_true[0], alpha=1.).item()
    result_pad = approxNDCGLoss(y_pred[1], y_true[1], alpha=1.).item()

    expected = -0.8499219417

    assert math.isfinite(result) and math.isfinite(result_pad)
    assert (result == approx(result_pad)) and (result == approx(expected))
