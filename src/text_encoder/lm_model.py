import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class HuggingFaceLM:
    def __init__(self, model_name, device="auto", batch_size=32):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = torch.device(device)
        self._batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._model = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
            .to(self._device)
            .eval()
        )

    @torch.inference_mode()
    def score(self, texts):
        log_probs = list()

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            inputs = self._tokenizer(
                [self._tokenizer.bos_token + text for text in batch],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            ).to(self._device)

            logits = self._model(**inputs, use_cache=False).logits.to(torch.float32)
            targets = inputs["input_ids"][:, 1:].clone()
            targets[
                inputs["attention_mask"][:, 1:] == 0
            ] = -100  # cross entropy ignore index

            losses = F.cross_entropy(
                logits[:, :-1].transpose(1, 2),
                targets,
                reduction="none",
            )
            log_probs.extend((-losses.sum(dim=1)).cpu().tolist())

        return log_probs
