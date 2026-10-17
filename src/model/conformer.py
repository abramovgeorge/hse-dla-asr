import torch.nn as nn

from src.model.conformer_encoder import ConformerEncoder


class ConformerCTC(nn.Module):
    """
    CTC model using audio encoder based on the Conformer paper.
    """

    def __init__(
        self,
        n_tokens,
        n_feats,
        hidden_size,
        num_layers,
        num_heads,
        conv_channels,
        conv_kernel_size,
        ff_expansion,
        dropout,
    ):
        """
        Args:
            n_tokens (int): number of tokens in the vocabulary, including blank.
            n_feats (int): number of input mel features.
            hidden_size (int): number of hidden features in the encoder.
            num_layers (int): number of Conformer layers.
            num_heads (int): number of attention heads.
            conv_channels (int): number of channels in subsampling convolutions.
            conv_kernel_size (int): odd kernel size in Conformer convolutions.
            ff_expansion (int): expansion factor in feed-forward modules.
            dropout (float): dropout probability.
        """
        super().__init__()

        self.encoder = ConformerEncoder(
            n_feats,
            hidden_size,
            num_layers,
            num_heads,
            conv_channels,
            conv_kernel_size,
            ff_expansion,
            dropout,
        )

        self.head = nn.Linear(hidden_size, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        x, log_probs_length = self.encoder(spectrogram, spectrogram_length)
        x = self.head(x)
        log_probs = nn.functional.log_softmax(x, dim=-1)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
