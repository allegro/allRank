from typing import Tuple

import allrank.models.losses as losses
import torch


def rank_listings(dataloader, model, device) -> Tuple[torch.Tensor, torch.Tensor]:
    reranked_X = []
    reranked_y = []
    model.eval()
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
            reranked_X.append(torch.gather(X, dim=1, index=indices_X))
            reranked_y.append(torch.gather(y_true, dim=1, index=indices))

    combined_X = torch.cat(reranked_X)
    combined_y = torch.cat(reranked_y)
    return combined_X, combined_y
