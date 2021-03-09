import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS
from allrank.models.losses.loss_utils import deterministic_neural_sort, sinkhorn_scaling, stochastic_neural_sort
from allrank.models.metrics import dcg
from allrank.models.model_utils import get_torch_device


def neuralNDCG(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, temperature=1., powered_relevancies=True, k=None,
               stochastic=False, n_samples=32, beta=0.1, log_scores=True):
    """
    NeuralNDCG loss introduced in "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable
    Relaxation of Sorting" - https://arxiv.org/abs/2102.07831. Based on the NeuralSort algorithm.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param temperature: temperature for the NeuralSort algorithm
    :param powered_relevancies: whether to apply 2^x - 1 gain function, x otherwise
    :param k: rank at which the loss is truncated
    :param stochastic: whether to calculate the stochastic variant
    :param n_samples: how many stochastic samples are taken, used if stochastic == True
    :param beta: beta parameter for NeuralSort algorithm, used if stochastic == True
    :param log_scores: log_scores parameter for NeuralSort algorithm, used if stochastic == True
    :return: loss value, a torch.Tensor
    """
    dev = get_torch_device()

    if k is None:
        k = y_true.shape[1]

    mask = (y_true == padded_value_indicator)
    # Choose the deterministic/stochastic variant
    if stochastic:
        P_hat = stochastic_neural_sort(y_pred.unsqueeze(-1), n_samples=n_samples, tau=temperature, mask=mask,
                                       beta=beta, log_scores=log_scores)
    else:
        P_hat = deterministic_neural_sort(y_pred.unsqueeze(-1), tau=temperature, mask=mask).unsqueeze(0)

    # Perform sinkhorn scaling to obtain doubly stochastic permutation matrices
    P_hat = sinkhorn_scaling(P_hat.view(P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3]),
                             mask.repeat_interleave(P_hat.shape[0], dim=0), tol=1e-6, max_iter=50)
    P_hat = P_hat.view(int(P_hat.shape[0] / y_pred.shape[0]), y_pred.shape[0], P_hat.shape[1], P_hat.shape[2])

    # Mask P_hat and apply to true labels, ie approximately sort them
    P_hat = P_hat.masked_fill(mask[None, :, :, None] | mask[None, :, None, :], 0.)
    y_true_masked = y_true.masked_fill(mask, 0.).unsqueeze(-1).unsqueeze(0)
    if powered_relevancies:
        y_true_masked = torch.pow(2., y_true_masked) - 1.

    ground_truth = torch.matmul(P_hat, y_true_masked).squeeze(-1)
    discounts = (torch.tensor(1.) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.)).to(dev)
    discounted_gains = ground_truth * discounts

    if powered_relevancies:
        idcg = dcg(y_true, y_true, ats=[k]).permute(1, 0)
    else:
        idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).permute(1, 0)

    discounted_gains = discounted_gains[:, :, :k]
    ndcg = discounted_gains.sum(dim=-1) / (idcg + DEFAULT_EPS)
    idcg_mask = idcg == 0.
    ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.)

    assert (ndcg < 0.).sum() >= 0, "every ndcg should be non-negative"
    if idcg_mask.all():
        return torch.tensor(0.)

    mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore
    return -1. * mean_ndcg  # -1 cause we want to maximize NDCG


def neuralNDCG_transposed(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, temperature=1.,
                          powered_relevancies=True, k=None, stochastic=False, n_samples=32, beta=0.1, log_scores=True,
                          max_iter=50, tol=1e-6):
    """
    NeuralNDCG Transposed loss introduced in "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable
    Relaxation of Sorting" - https://arxiv.org/abs/2102.07831. Based on the NeuralSort algorithm.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param temperature: temperature for the NeuralSort algorithm
    :param powered_relevancies: whether to apply 2^x - 1 gain function, x otherwise
    :param k: rank at which the loss is truncated
    :param stochastic: whether to calculate the stochastic variant
    :param n_samples: how many stochastic samples are taken, used if stochastic == True
    :param beta: beta parameter for NeuralSort algorithm, used if stochastic == True
    :param log_scores: log_scores parameter for NeuralSort algorithm, used if stochastic == True
    :param max_iter: maximum iteration count for Sinkhorn scaling
    :param tol: tolerance for Sinkhorn scaling
    :return: loss value, a torch.Tensor
    """
    dev = get_torch_device()

    if k is None:
        k = y_true.shape[1]

    mask = (y_true == padded_value_indicator)

    if stochastic:
        P_hat = stochastic_neural_sort(y_pred.unsqueeze(-1), n_samples=n_samples, tau=temperature, mask=mask,
                                       beta=beta, log_scores=log_scores)
    else:
        P_hat = deterministic_neural_sort(y_pred.unsqueeze(-1), tau=temperature, mask=mask).unsqueeze(0)

    # Perform sinkhorn scaling to obtain doubly stochastic permutation matrices
    P_hat_masked = sinkhorn_scaling(P_hat.view(P_hat.shape[0] * y_pred.shape[0], y_pred.shape[1], y_pred.shape[1]),
                                    mask.repeat_interleave(P_hat.shape[0], dim=0), tol=tol, max_iter=max_iter)
    P_hat_masked = P_hat_masked.view(P_hat.shape[0], y_pred.shape[0], y_pred.shape[1], y_pred.shape[1])
    discounts = (torch.tensor(1) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.)).to(dev)

    # This takes care of the @k metric truncation - if something is @>k, it is useless and gets 0.0 discount
    discounts[k:] = 0.
    discounts = discounts[None, None, :, None]

    # Here the discounts become expected discounts
    discounts = torch.matmul(P_hat_masked.permute(0, 1, 3, 2), discounts).squeeze(-1)
    if powered_relevancies:
        gains = torch.pow(2., y_true) - 1
        discounted_gains = gains.unsqueeze(0) * discounts
        idcg = dcg(y_true, y_true, ats=[k]).squeeze()
    else:
        gains = y_true
        discounted_gains = gains.unsqueeze(0) * discounts
        idcg = dcg(y_true, y_true, ats=[k]).squeeze()

    ndcg = discounted_gains.sum(dim=2) / (idcg + DEFAULT_EPS)
    idcg_mask = idcg == 0.
    ndcg = ndcg.masked_fill(idcg_mask, 0.)

    assert (ndcg < 0.).sum() >= 0, "every ndcg should be non-negative"
    if idcg_mask.all():
        return torch.tensor(0.)

    mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore
    return -1. * mean_ndcg  # -1 cause we want to maximize NDCG
