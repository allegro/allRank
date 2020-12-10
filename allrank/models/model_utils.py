from typing import Any

import numpy as np
import torch
import torch.nn as nn

from allrank.utils.file_utils import is_gs_path, copy_file_to_local
from allrank.utils.ltr_logging import get_logger

logger = get_logger()


def get_torch_device():
    """
    Getter for an available pyTorch device.
    :return: CUDA-capable GPU if available, CPU otherwise
    """
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_num_params(model: nn.Module) -> int:
    """
    Calculation of the number of nn.Module parameters.
    :param model: nn.Module
    :return: number of parameters
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params  # type: ignore


def log_num_params(num_params: int) -> None:
    """
    Logging num_params to the global logger.
    :param num_params: number of parameters to log
    """
    logger.info("Model has {} trainable parameters".format(num_params))


class CustomDataParallel(nn.DataParallel):
    """
    Wrapper for scoring with nn.DataParallel object containing LTRModel.
    """

    def score(self, x, mask, indices):
        """
        Wrapper function for a forward pass through the whole LTRModel and item scoring.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: scores of shape [batch_size, slate_length]
        """
        return self.module.score(x, mask, indices)  # type: ignore


def load_state_dict_from_file(path: str, device: Any):
    if is_gs_path(path):
        path = copy_file_to_local(path)

    return torch.load(path, map_location=device)
