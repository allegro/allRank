from typing import Tuple

from torch.utils.data.dataloader import DataLoader

import allrank.models.losses as losses
import torch

from allrank.data.dataset_loading import LibSVMDataset, create_data_loaders
from allrank.models.model_utils import get_torch_device


def rank_listings(train_ds: LibSVMDataset, val_ds: LibSVMDataset, model, config) \
        -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

    train_dl, val_dl = create_data_loaders(
        train_ds, val_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)
    train_listings = __rank_listings(train_dl, model)
    val_listings = __rank_listings(val_dl, model)
    return train_listings, val_listings


def __rank_listings(dataloader: DataLoader, model) -> Tuple[torch.Tensor, torch.Tensor]:
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
