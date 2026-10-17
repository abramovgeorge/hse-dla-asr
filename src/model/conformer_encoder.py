import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    # Adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, max_length, embed_dim, dropout):
        super().__init__()

        assert (
            embed_dim % 2 == 0
        ), "Positional encoding implementation works only with even hidden size (at least for now)"

        self.pos_features = torch.zeros(max_length, embed_dim)

        positions = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        freqs = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000) / embed_dim)
        ).unsqueeze(0)

        arguments = positions * freqs
        self.pos_features[:, 0::2] = torch.sin(arguments)
        self.pos_features[:, 1::2] = torch.cos(arguments)
        self.pos_features = self.pos_features.unsqueeze(0)
        self.pos_features = nn.Parameter(self.pos_features, requires_grad=False)
        # pos_features: (1, max_length, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        outputs = inputs + self.pos_features[:, : inputs.shape[1]]
        return self.dropout(outputs)


class ConformerLayer(nn.Module):
    """
    Layer of the model from the Conformer paper.
    """

    class Transpose(nn.Module):
        """
        Helper module for Conformer convolution block
        """

        def __init__(self, dim0, dim1):
            super().__init__()
            self.dim0 = dim0
            self.dim1 = dim1

        def forward(self, x):
            return x.transpose(self.dim0, self.dim1)

    def __init__(self, hidden_size, num_heads, ff_expansion, conv_kernel_size, dropout):
        """
        Args:
            hidden_size (int): number of input and output features.
                Must be divisible by num_heads.
            num_heads (int): number of attention heads.
            ff_expansion (int): expansion factor in feed-forward modules.
            conv_kernel_size (int): odd kernel size of the depthwise convolution.
            dropout (float): dropout probability.
        """
        super().__init__()

        assert (
            conv_kernel_size % 2 == 1
        ), "Kernel size should be odd for formulas to work"

        self.ffn1 = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * ff_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * ff_expansion, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * ff_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * ff_expansion, hidden_size),
            nn.Dropout(dropout),
        )

        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)

        self.conv1 = nn.Sequential(
            nn.LayerNorm(hidden_size),
            self.Transpose(1, 2),
            nn.Conv1d(hidden_size, 2 * hidden_size, 1),
            nn.GLU(1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                hidden_size,
                hidden_size,
                conv_kernel_size,
                padding=(conv_kernel_size - 1) // 2,
                groups=hidden_size,
            ),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Conv1d(hidden_size, hidden_size, 1),
            self.Transpose(1, 2),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, padding_mask):
        """
        Args:
            x (Tensor): input embeddings of size (B, T, D).
            padding_mask (Tensor): boolean mask of size (B, T).
                True marks padded positions.
        Returns:
            Tensor: output embeddings of size (B, T, D), with padding zeroed.
        """
        x = x + 0.5 * self.ffn1(x)
        z = self.attn_norm(x)
        attn_out, _ = self.attn(
            z,
            z,
            z,
            padding_mask,
            need_weights=False,
        )
        attn_out = self.attn_dropout(attn_out)
        x = x + attn_out
        conv_out = self.conv1(x)
        conv_out = conv_out.masked_fill(padding_mask.unsqueeze(1), 0.0)
        conv_out = self.conv2(conv_out)
        x = x + conv_out
        x = x + 0.5 * self.ffn2(x)
        x = self.norm(x)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


class ConformerEncoder(nn.Module):
    """
    Audio encoder based on the Conformer paper.
    """

    def __init__(
        self,
        n_feats,
        hidden_size,
        num_layers,
        num_heads,
        conv_channels,
        conv_kernel_size,
        ff_expansion,
        dropout,
        max_len=10000,
    ):
        """
        Args:
            n_feats (int): number of input mel features.
            hidden_size (int): even number of hidden features.
                Must be divisible by num_heads.
            num_layers (int): number of Conformer layers.
            num_heads (int): number of attention heads in each layer.
            conv_channels (int): number of channels in subsampling convolutions.
            conv_kernel_size (int): odd kernel size in Conformer convolutions.
            ff_expansion (int): expansion factor in feed-forward modules.
            dropout (float): dropout probability.
            max_len (int): maximum sequence length after subsampling.
        """
        super().__init__()

        assert (
            hidden_size % num_heads == 0
        ), "Hidden size should be divisible by the number of heads"

        self.downsampler = nn.Sequential(
            nn.Conv2d(1, conv_channels, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(conv_channels, conv_channels, 3, 2, 1),
            nn.LeakyReLU(),
        )

        for _ in range(2):
            n_feats = (n_feats + 1) // 2
        self.downsample_mlp = nn.Linear(conv_channels * n_feats, hidden_size)

        self.layers = nn.ModuleList(
            [
                ConformerLayer(
                    hidden_size, num_heads, ff_expansion, conv_kernel_size, dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.pe = SinusoidalPositionalEncoding(max_len, hidden_size, dropout)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            Tuple[Tensor, Tensor]: output embeddings of size (B, T', D) and length of size (B)
        """
        x = self._downsample_inputs(spectrogram.unsqueeze(1))  # (B, T', D)
        length = self._transform_input_lengths(spectrogram_length).to(x.device)
        positions = torch.arange(x.shape[1], device=x.device)
        padding_mask = positions.unsqueeze(0) >= length.unsqueeze(1)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x, length

    def _downsample_inputs(self, spectrogram):
        """
        Args:
            spectrogram (Tensor): original spectrogram
        Returns:
            Tensor: downsampled spectrogram
        """
        x = self.downsampler(spectrogram)  # (B, C, F', T')
        x = x.permute(0, 3, 1, 2)  # (B, T', C, F')
        x = x.flatten(start_dim=2)  # (B, T', C x F')
        x = self.downsample_mlp(x)  # (B, T', D)
        return x

    def _transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        lengths = input_lengths

        for layer in self.downsampler:
            if isinstance(layer, nn.Conv2d):
                k = layer.kernel_size[-1]
                s = layer.stride[-1]
                p = layer.padding[-1]
                d = layer.dilation[-1]

                # formula from https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html
                lengths = (lengths + 2 * p - d * (k - 1) - 1) // s + 1

        return lengths
