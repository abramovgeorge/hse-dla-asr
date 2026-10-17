import torch
from audiomentations import AddBackgroundNoise
from torch import Tensor, nn


class BackgroundNoise(nn.Module):
    def __init__(self, sample_rate, *args, **kwargs):
        super().__init__()
        self._sample_rate = sample_rate
        self._aug = AddBackgroundNoise(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.detach().cpu().squeeze(0).numpy()
        x = self._aug(x, sample_rate=self._sample_rate)
        return torch.from_numpy(x).unsqueeze(0).to(data)
