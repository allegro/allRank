import torch
import torch.nn.functional as F

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS


def listNet(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):  # dimensions: [batch, listing]

    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))
