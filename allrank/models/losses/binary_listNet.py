import torch
import torch.nn.functional as F

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS


def binary_listNet(y_pred, y_true, eps=DEFAULT_EPS,
                   padded_value_indicator=PADDED_Y_VALUE):  # dimensions: [batch, listing]

    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0
    normalizer = torch.unsqueeze(y_true.sum(dim=-1), 1)
    normalizer[normalizer == 0.0] = 1.0
    normalizer = normalizer.expand(-1, y_true.shape[1])
    y_true = torch.div(y_true, normalizer)

    preds_smax = F.softmax(y_pred, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(y_true * preds_log, dim=1))
