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
    texts = list()
    audio_paths = list()

    for item in dataset_items:
        texts.append(item["text"])
        audio_paths.append(item["audio_path"])

    for field in (
        "audio",
        "spectrogram",
        "audio_raw",
        "spectrogram_raw",
        "text_encoded",
    ):
        lens = [item[field].shape[-1] for item in dataset_items]
        max_len = max(lens)
        items = list()

        for item, length in zip(dataset_items, lens):
            data = item[field]
            padding = torch.zeros((*data.shape[:-1], max_len - length))
            items.append(torch.concat([data, padding], dim=-1))

        batch[field] = torch.stack(items).squeeze(1)
        batch[f"{field}_length"] = torch.tensor(lens)

    batch["text_encoded"] = batch["text_encoded"].long()

    batch["text"] = texts
    batch["audio_path"] = audio_paths

    return batch
