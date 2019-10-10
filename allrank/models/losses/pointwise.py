import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE


def pointwise_rmse(y_pred, y_true, no_of_levels,
                   padded_value_indicator=PADDED_Y_VALUE):  # dimensions: [batch, listing]

    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    valid_mask = (y_true != padded_value_indicator).type(torch.float32)

    y_true[mask] = 0
    y_pred[mask] = 0

    errors = (y_true - no_of_levels * y_pred)

    squared_errors = errors ** 2

    mean_squared_errors = torch.sum(squared_errors, dim=1) / torch.sum(valid_mask, dim=1)

    rmses = torch.sqrt(mean_squared_errors)

    return torch.mean(rmses)
