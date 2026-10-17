import re
from collections import defaultdict
from string import ascii_lowercase

import numpy as np
import torch

# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, lm_model=None, lm_weight=0.1, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
            lm_model (HuggingFaceLM | None): LM for shallow fusion during beam search
            lm_weight (float): weight for LM log probs during shallow fusion
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self._lm_model = lm_model
        self._lm_weight = lm_weight

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
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
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        """
        Decode the CTC indices.
        Args:
            inds(list): list of tokens
        Returns:
            raw_text (str): raw text without empty tokens and repetitions.
        """
        if len(inds) == 0:
            return ""
        no_reps = [inds[0]]
        for i in range(1, len(inds)):
            if inds[i] != inds[i - 1]:
                no_reps.append(inds[i])
        no_blanks = [ind for ind in no_reps if ind != self.char2ind[self.EMPTY_TOK]]
        return "".join([self.ind2char[int(ind)] for ind in no_blanks]).strip()

    def ctc_beam_search(self, log_probs, beam_size=10):
        """
        Decode the CTC output using beam search.
        Args:
            log_probs (Tensor): log probabilities of shape [T, vocab_size].
            beam_size (int): number of beams to keep at each step.
        Returns:
            list: (text, log_probability) pairs sorted by probability.
        """
        log_probs = log_probs.detach().cpu().numpy()
        empty_ind = self.char2ind[self.EMPTY_TOK]
        beam = {"": [0.0, -np.inf]}

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

                    char = self.ind2char[ind]
                    if len(prefix) > 0 and prefix[-1] == char:
                        new_beam[prefix][1] = np.logaddexp(
                            new_beam[prefix][1], log_p_nb + log_p
                        )
                        new_beam[prefix + char][1] = np.logaddexp(
                            new_beam[prefix + char][1], log_p_b + log_p
                        )
                    else:
                        new_beam[prefix + char][1] = np.logaddexp(
                            new_beam[prefix + char][1],
                            np.logaddexp(log_p_b, log_p_nb) + log_p,
                        )

            beam = dict(
                sorted(
                    new_beam.items(),
                    key=lambda x: -(np.logaddexp(x[1][0], x[1][1])),
                )[:beam_size]
            )

        results = [(prefix, np.logaddexp(p[0], p[1])) for prefix, p in beam.items()]

        if self._lm_model is not None:
            lm_scores = self._lm_model.score([text for text, _ in results])

            results = [
                (text, old_score + self._lm_weight * lm_score)
                for (text, old_score), lm_score in zip(results, lm_scores)
            ]

        return sorted(
            results,
            key=lambda x: -x[1],
        )[:beam_size]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
