import torch

from allrank.models.losses.neuralNDCG import neuralNDCG, neuralNDCG_transposed
from allrank.models.metrics import ndcg


def neuralNDCG_wrap(
        y_pred, y_true, temperature=1e-4, powered_relevancies=True, k=None, stochastic=False,
        n_samples=1024, beta=0.001, transposed=False):
    if transposed:
        fun = neuralNDCG_transposed  # type: ignore
    else:
        fun = neuralNDCG  # type: ignore

    return fun(torch.tensor([y_pred]), torch.tensor([y_true]), temperature=temperature,
               powered_relevancies=powered_relevancies, k=k,
               stochastic=stochastic, n_samples=n_samples, beta=beta).item()


def ndcg_wrap(y_pred, y_true, ats=None):
    return ndcg(torch.tensor([y_pred]), torch.tensor([y_true]), ats=ats).mean().item()
