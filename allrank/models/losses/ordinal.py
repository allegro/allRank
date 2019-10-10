import torch
from torch.nn import BCELoss

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_torch_device


def with_ordinals(y, n):
    # y dimensions: [batch, listing]
    # output dimensions: [batch, listing, n (number of ordinal values]
    dev = get_torch_device()
    one_to_n = torch.arange(start=1, end=n + 1, dtype=torch.float, device=dev)
    unsqueezed = y.unsqueeze(2).repeat(1, 1, n)
    mask = unsqueezed == PADDED_Y_VALUE
    ordinals = (unsqueezed >= one_to_n).type(torch.float)
    ordinals[mask] = PADDED_Y_VALUE
    return ordinals


def ordinal(y_pred, y_true, n,
            padded_value_indicator=PADDED_Y_VALUE):  # dimensions: [batch, listing]

    dev = get_torch_device()

    y_pred = y_pred.clone()
    y_true = with_ordinals(y_true.clone(), n)

    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    ls = BCELoss(reduction='none')(y_pred, y_true)
    ls[mask] = 0.0

    document_loss = torch.sum(ls, dim=2)
    sum_valid = torch.sum(valid_mask, dim=2).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=dev)

    loss_output = torch.sum(document_loss) / torch.sum(sum_valid)

    return loss_output
