import torch
from torch.nn import BCELoss

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_torch_device


def bce(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE):
    dev = get_torch_device()

    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    ls = BCELoss(reduction='none')(y_pred, y_true)
    ls[mask] = 0.0

    document_loss = torch.sum(ls, dim=-1)
    sum_valid = torch.sum(valid_mask, dim=-1).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=dev)

    loss_output = torch.sum(document_loss) / torch.sum(sum_valid)

    return loss_output
