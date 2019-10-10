import numpy as np
import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_torch_device


def ndcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1):
    # y_true and y_pred have dimensions [listing, document_score]
    # returns a tensor with ndcg values at positions specified by 'ats' with dimensions [listing, dcg_at]

    idcg = dcg(y_true, y_true, ats, gain_function)
    ndcg_ = dcg(y_pred, y_true, ats, gain_function) / idcg
    idcg_mask = idcg == 0
    ndcg_[idcg_mask] = 0.  # if idcg == 0 , set ndcg to 0

    assert (ndcg_ < 0.0).sum() >= 0, "every ndcg should be non-negative"

    return ndcg_


def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true):
    mask = y_true == PADDED_Y_VALUE

    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)


def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1):
    # y_true and y_pred have dimensions [listing, document_score]
    # returns a tensor with ndcg values at positions specified by 'ats' with dimensions [listing, dcg_at]
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true)

    dev = get_torch_device()

    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0)).to(
        device=dev)

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, :np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, ats_tensor]

    return dcg


def mrr(y_pred, y_true, ats=None):
    # y_true and y_pred have dimensions [listing, document_score]
    # returns a tensor with mrr values at positions specified by 'ats' with dimensions [listing, mrr_at]
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    if ats is None:
        ats = [y_true.shape[1]]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true)

    values, indices = torch.max(true_sorted_by_preds, dim=1)
    indices = indices.type_as(values).unsqueeze(dim=0).t().expand(len(y_true), len(ats))

    dev = get_torch_device()

    ats_rep = torch.tensor(data=ats, device=dev, dtype=torch.float32).expand(len(y_true), len(ats))

    within_at_mask = (indices < ats_rep).type(torch.float32)

    result = torch.tensor(1.0) / (indices + torch.tensor(1.0))

    zero_sum_mask = torch.sum(values) == 0.0
    result[zero_sum_mask] = 0.0

    result = result * within_at_mask

    return result
