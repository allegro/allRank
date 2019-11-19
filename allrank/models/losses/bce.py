import torch
from torch.nn import BCELoss

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_torch_device


def bce(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE):
    """
    Binary Cross-Entropy loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    device = get_torch_device()

    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    ls = BCELoss(reduction='none')(y_pred, y_true)
    ls[mask] = 0.0

    document_loss = torch.sum(ls, dim=-1)
    sum_valid = torch.sum(valid_mask, dim=-1).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=device)

    loss_output = torch.sum(document_loss) / torch.sum(sum_valid)

    return loss_output
