import torch
from librosa.effects import time_stretch
from torch import Tensor, nn


class TimeStretch(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self._rate = rate

    def __call__(self, data: Tensor):
        x = data.detach().cpu().squeeze(0).numpy()
        x = time_stretch(x, rate=self._rate)
        return torch.from_numpy(x).unsqueeze(0).to(data)
