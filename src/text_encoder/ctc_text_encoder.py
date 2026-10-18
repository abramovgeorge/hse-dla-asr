import os
import re
from collections import defaultdict
from pathlib import Path
from string import ascii_lowercase

import numpy as np
import torch
from joblib import Parallel, delayed
from tokenizers import CharBPETokenizer, Tokenizer

# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(
        self,
        alphabet=None,
        tokenizer_path=None,
        tokenizer_text_dir=None,
        vocab_size=256,
        lm_model=None,
        lm_weight=0.1,
        beam_search_njobs=1,
        **kwargs,
    ):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
            tokenizer_path (str | None): if not None,
                path to save/load tokenizer from.
            tokenizer_text_dir (str | None): if not None,
                path to load texts for training the tokenizer.
            vocab_size (int): vocab size for tokenizer.
            lm_model (HuggingFaceLM | None): LM for fusion during beam search
            lm_weight (float): weight for LM log probs during fusion
            beam_search_njobs (int): number of jobs for parallel beam search
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self._tokenizer = None
        if tokenizer_path is not None:
            self._tokenizer = self._load_or_train_tokenizer(
                tokenizer_path, tokenizer_text_dir, vocab_size
            )
            self.vocab = [self.EMPTY_TOK] + [
                self._tokenizer.id_to_token(i)
                for i in range(self._tokenizer.get_vocab_size())
            ]

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self._lm_model = lm_model
        self._lm_weight = lm_weight

        self._beam_search_njobs = beam_search_njobs

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        if self._tokenizer is not None:
            ids = self._tokenizer.encode(text).ids
            return torch.tensor(
                [i + 1 for i in ids],
                dtype=torch.long,
            ).unsqueeze(0)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        if self._tokenizer is not None:
            return self._tokenizer.decode([ind - 1 for ind in inds if ind != 0]).strip()
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        """
        Decode the CTC indices.
        Args:
            inds(list): list of tokens
        Returns:
            raw_text (str): raw text without empty tokens and repetitions.
        """
        inds = [int(i) for i in inds]  # inds is tensor for bpe
        if len(inds) == 0:
            return ""
        no_reps = [inds[0]]
        for i in range(1, len(inds)):
            if inds[i] != inds[i - 1]:
                no_reps.append(inds[i])
        return self.decode(
            [ind for ind in no_reps if ind != self.char2ind[self.EMPTY_TOK]]
        )

    def ctc_beam_search(self, log_probs, beam_size=10):
        """
        Decode the CTC output using beam search.
        Args:
            log_probs (Tensor): log probabilities of shape [T, vocab_size].
            beam_size (int): number of beams to keep at each step.
        Returns:
            list: (text, log_probability) pairs sorted by probability.
        """
        return self.ctc_beam_search_batched(
            log_probs.unsqueeze(0), torch.tensor([log_probs.shape[0]]), beam_size
        )[0]

    def ctc_beam_search_batched(self, log_probs, log_probs_length, beam_size=10):
        # log_probs: (B, T, V)
        log_probs = log_probs.detach().cpu().numpy()
        log_probs_length = log_probs_length.detach().cpu().numpy()
        beam_jobs = Parallel(n_jobs=self._beam_search_njobs, backend="loky")(
            delayed(CTCTextEncoder._ctc_beam_dp)(
                probs[:length], beam_size, self.char2ind[self.EMPTY_TOK]
            )
            for probs, length in zip(log_probs, log_probs_length)
        )
        results = [
            [(self.decode(prefix), score) for prefix, score in beams]
            for beams in beam_jobs
        ]
        if self._lm_model is not None:
            lm_scores = iter(
                self._lm_model.score([text for beams in results for text, _ in beams])
            )

            results = [
                [
                    (text, old_score + self._lm_weight * next(lm_scores))
                    for text, old_score in beams
                ]
                for beams in results
            ]

        return [
            sorted(
                beams,
                key=lambda x: -x[1],
            )[:beam_size]
            for beams in results
        ]

    @staticmethod
    def _ctc_beam_dp(log_probs, beam_size, empty_ind):
        beam = {(): [0.0, -np.inf]}

        for log_prob_t in log_probs:
            new_beam = defaultdict(lambda: [-np.inf, -np.inf])
            for prefix, (log_p_b, log_p_nb) in beam.items():
                for ind, log_p in enumerate(log_prob_t):
                    if ind == empty_ind:
                        new_beam[prefix][0] = np.logaddexp(
                            new_beam[prefix][0],
                            np.logaddexp(log_p_b, log_p_nb) + log_p,
                        )
                        continue

                    new_prefix = prefix + (ind,)
                    if len(prefix) > 0 and prefix[-1] == ind:
                        new_beam[prefix][1] = np.logaddexp(
                            new_beam[prefix][1], log_p_nb + log_p
                        )
                        new_beam[new_prefix][1] = np.logaddexp(
                            new_beam[new_prefix][1], log_p_b + log_p
                        )
                    else:
                        new_beam[new_prefix][1] = np.logaddexp(
                            new_beam[new_prefix][1],
                            np.logaddexp(log_p_b, log_p_nb) + log_p,
                        )

            beam = dict(
                sorted(
                    new_beam.items(),
                    key=lambda x: -(np.logaddexp(x[1][0], x[1][1])),
                )[:beam_size]
            )

        return [(prefix, np.logaddexp(p[0], p[1])) for prefix, p in beam.items()]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    def _load_or_train_tokenizer(self, tokenizer_path, tokenizer_text_dir, vocab_size):
        tokenizer_path = Path(tokenizer_path)
        if tokenizer_path.exists():
            return Tokenizer.from_file(str(tokenizer_path))

        assert (
            tokenizer_text_dir is not None
        ), "Path to load texts for training the tokenizer is required"

        texts = []
        for dirpath, dirnames, filenames in os.walk(str(tokenizer_text_dir)):
            for trans_path in Path(dirpath).glob("*.trans.txt"):
                with trans_path.open() as f:
                    for line in f:
                        text = " ".join(line.split()[1:]).strip()
                        texts.append(self.normalize_text(text))

        assert len(texts) > 0, "No texts found"

        tokenizer = CharBPETokenizer(unk_token="[UNK]")
        tokenizer.train_from_iterator(
            texts,
            vocab_size=vocab_size,
            special_tokens=["[UNK]"],
        )

        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        return tokenizer
