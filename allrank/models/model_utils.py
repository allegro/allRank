import torch

import numpy as np
import torch.nn as nn

from allrank.utils.ltr_logging import get_logger

logger = get_logger()


def get_torch_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_num_params(model: nn.Module) -> int:
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def log_num_params(num_params: int) -> None:
    logger.info("Model has {} trainable parameters".format(num_params))


class CustomDataParallel(nn.DataParallel):
    def score(self, x, mask, indices):
        return self.module.score(x, mask, indices)
