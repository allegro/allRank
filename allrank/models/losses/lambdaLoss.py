import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS

"""Here are the NDCGLoss1 and NDCGLoss2(++) losses from The LambdaLoss Framework for Ranking Metric Optimization.
Please note that we are sticking to array operations most of the time, only applying a padding mask at the very end.
The code is therefore close to the original formulations but multiple clamps were needed not to alarm PyTorch anomaly checks.

We use notation from the original paper, hence the iDCG is maxDCG etc.
"""


def lambdaLoss(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, weighing_scheme=None, k=5, sigma=1., mu=10.,
               reduction="sum", reduction_log="binary"):
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)

    if weighing_scheme != "ndcgLoss1_scheme":
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    ndcg_at_k_idxs = torch.arange(y_pred.shape[1], device=device) < k
    ndcg_at_k_mask = ndcg_at_k_idxs[:, None] | ndcg_at_k_idxs[None, :]

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per listing.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=DEFAULT_EPS)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
    if weighing_scheme is None:
        weights = 1.
    else:
        weights = WEIGHING_SCHEMES[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
    scores_diffs[torch.isnan(scores_diffs)] = 0.
    weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=DEFAULT_EPS) ** weights).clamp(min=DEFAULT_EPS)
    if reduction_log == "natural":
        losses = torch.log(weighted_probas)
    elif reduction_log == "binary":
        losses = torch.log2(weighted_probas)
    else:
        raise ValueError("Reduction logarithm base can be either natural or binary")

    masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
    if reduction == "sum":
        loss = -torch.sum(masked_losses)
    elif reduction == "mean":
        loss = -torch.mean(masked_losses)
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss


def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
    deltas.diagonal().zero_()

    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lamdbaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(G[:, :, None] - G[:, None, :])


def ndcgLoss2PP_scheme(G, D, *args):
    return args[0] * ndcgLoss2_scheme(G, D) + lamdbaRank_scheme(G, D)


def rankNet_scheme(G, D, *args):
    return 1.


def rankNetWeightedByGTDiff_scheme(G, D, *args):
    return torch.abs(args[1][:, :, None] - args[1][:, None, :])


def rankNetWeightedByGTDiffPowed_scheme(G, D, *args):
    return torch.abs(torch.pow(args[1][:, :, None], 2) - torch.pow(args[1][:, None, :], 2))


WEIGHING_SCHEMES = {
    "ndcgLoss1_scheme": ndcgLoss1_scheme,
    "ndcgLoss2_scheme": ndcgLoss2_scheme,
    "lamdbaRank_scheme": lamdbaRank_scheme,
    "ndcgLoss2PP_scheme": ndcgLoss2PP_scheme,
    "rankNet_scheme": rankNet_scheme,
    "rankNetWeightedByGTDiff_scheme": rankNetWeightedByGTDiff_scheme,
    "rankNetWeightedByGTDiffPowed_scheme": rankNetWeightedByGTDiffPowed_scheme
}
