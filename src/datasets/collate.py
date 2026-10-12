import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    batch = dict()
    max_spectrogram_len = max(item["spectrogram"].shape[-1] for item in dataset_items)
    max_text_len = max(item["text_encoded"].shape[-1] for item in dataset_items)
    max_audio_len = max(item["audio"].shape[-1] for item in dataset_items)

    spectrograms = list()
    texts_encoded = list()
    audios = list()

    spectrogram_lens = list()
    text_encoded_lens = list()
    audio_lens = list()

    texts = list()
    audio_paths = list()

    for item in dataset_items:
        spectrogram = item["spectrogram"]
        spectrograms.append(
            torch.concat(
                [
                    spectrogram,
                    torch.zeros(
                        (
                            1,
                            spectrogram.shape[1],
                            max_spectrogram_len - spectrogram.shape[-1],
                        )
                    ),
                ],
                dim=-1,
            )
        )
        spectrogram_lens.append(spectrogram.shape[-1])

        text_encoded = item["text_encoded"]
        texts_encoded.append(
            torch.concat(
                [text_encoded, torch.zeros((1, max_text_len - text_encoded.shape[-1]))],
                dim=-1,
            )
        )
        text_encoded_lens.append(text_encoded.shape[-1])

        audio = item["audio"]
        audios.append(
            torch.concat(
                [audio, torch.zeros((1, max_audio_len - audio.shape[-1]))], dim=-1
            )
        )
        audio_lens.append(audio.shape[-1])

        texts.append(item["text"])
        audio_paths.append(item["audio_path"])

    batch["spectrogram"] = torch.stack(spectrograms).squeeze(1)
    batch["spectrogram_length"] = torch.tensor(spectrogram_lens)

    batch["text_encoded"] = torch.stack(texts_encoded).to(torch.long).squeeze(1)
    batch["text_encoded_length"] = torch.tensor(text_encoded_lens)

    batch["audio"] = torch.stack(audios).squeeze(1)
    batch["audio_length"] = torch.tensor(audio_lens)

    batch["text"] = texts
    batch["audio_path"] = audio_paths

    return batch
