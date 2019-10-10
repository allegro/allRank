import math

import torch
from pytest import approx

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses.lambdaLoss import lambdaLoss


def test_ndcgloss1_ignores_padded():
    y_pred = [torch.tensor([[0.5, 0.3, 0.5]]), torch.tensor([[0.5, 0.3, 0.5, 1.0]])]
    y_true = [torch.tensor([[0.5, 0.3, 0.5]]), torch.tensor([[0.5, 0.3, 0.5, PADDED_Y_VALUE]])]

    result = lambdaLoss(y_pred[0], y_true[0], weighing_scheme="ndcgLoss1_scheme", reduction_log="binary").item()
    result_pad = lambdaLoss(y_pred[1], y_true[1], weighing_scheme="ndcgLoss1_scheme", reduction_log="binary").item()

    expected = 2.9272110462

    assert math.isfinite(result) and math.isfinite(result_pad)
    assert (result == approx(result_pad)) and (result == approx(expected))


def test_ndcgloss2_ignores_padded():
    y_pred = [torch.tensor([[0.5, 0.3, 0.5]]), torch.tensor([[0.5, 0.3, 0.5, 1.0]])]
    y_true = [torch.tensor([[0.5, 0.3, 0.5]]), torch.tensor([[0.5, 0.3, 0.5, PADDED_Y_VALUE]])]

    result = lambdaLoss(y_pred[0], y_true[0], weighing_scheme="ndcgLoss2PP_scheme", reduction_log="binary").item()
    result_pad = lambdaLoss(y_pred[1], y_true[1], weighing_scheme="ndcgLoss2PP_scheme", reduction_log="binary").item()

    expected = 1.1244146823

    assert math.isfinite(result) and math.isfinite(result_pad)
    assert (result == approx(result_pad)) and (result == approx(expected))


def test_ranknet_ignores_padded():
    y_pred = [torch.tensor([[0.5, 0.3, 0.5]]), torch.tensor([[0.5, 0.3, 0.5, 1.0]])]
    y_true = [torch.tensor([[0.5, 0.3, 0.5]]), torch.tensor([[0.5, 0.3, 0.5, PADDED_Y_VALUE]])]

    result = lambdaLoss(y_pred[0], y_true[0], weighing_scheme="rankNet_scheme", reduction_log="natural").item()
    result_pad = lambdaLoss(y_pred[1], y_true[1], weighing_scheme="rankNet_scheme", reduction_log="natural").item()

    expected = 1.1962778568

    assert math.isfinite(result) and math.isfinite(result_pad)
    assert (result == approx(result_pad)) and (result == approx(expected))
