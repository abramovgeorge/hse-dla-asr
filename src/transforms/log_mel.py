import torch
from torch import Tensor, nn


class LogMel(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self._eps = eps

    def forward(self, data: Tensor):
        return torch.log(torch.clip(data, self._eps))
