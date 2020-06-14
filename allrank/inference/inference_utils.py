from typing import Tuple, Dict

import torch
from torch.utils.data.dataloader import DataLoader

import allrank.models.losses as losses
from allrank.config import Config
from allrank.data.dataset_loading import LibSVMDataset
from allrank.models.model import LTRModel
from allrank.models.model_utils import get_torch_device


def rank_slates(datasets: Dict[str, LibSVMDataset], model: LTRModel, config: Config) \
        -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Ranks given datasets according to a given model

    :param datasets: dictionary of role -> dataset that will be ranked
    :param model: a model to use for scoring documents
    :param config: config for DataLoaders
    :return: dictionary of role -> ranked dataset
        every dataset is a Tuple of torch.Tensor - storing X and y in the descending order of the scores.
    """

    dataloaders = {role: __create_data_loader(ds, config) for role, ds in datasets.items()}

    ranked_slates = {role: __rank_slates(dl, model) for role, dl in dataloaders.items()}

    return ranked_slates


def __create_data_loader(ds: LibSVMDataset, config: Config) -> DataLoader:
    return DataLoader(ds, batch_size=config.data.batch_size, num_workers=config.data.num_workers, shuffle=False)


def __rank_slates(dataloader: DataLoader, model: LTRModel) -> Tuple[torch.Tensor, torch.Tensor]:
    reranked_X = []
    reranked_y = []
    model.eval()
    device = get_torch_device()
    with torch.no_grad():
        for xb, yb, _ in dataloader:
            X = xb.type(torch.float32).to(device=device)
            y_true = yb.to(device=device)

            input_indices = torch.ones_like(y_true).type(torch.long)
            mask = (y_true == losses.PADDED_Y_VALUE)
            scores = model.score(X, mask, input_indices)

            scores[mask] = float('-inf')

            _, indices = scores.sort(descending=True, dim=-1)
            indices_X = torch.unsqueeze(indices, -1).repeat_interleave(X.shape[-1], -1)
            reranked_X.append(torch.gather(X, dim=1, index=indices_X).cpu())
            reranked_y.append(torch.gather(y_true, dim=1, index=indices).cpu())

    combined_X = torch.cat(reranked_X)
    combined_y = torch.cat(reranked_y)
    return combined_X, combined_y
