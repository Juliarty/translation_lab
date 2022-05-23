from typing import List, Tuple

import torch.utils.data
from torch import Tensor
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab

from src.params.dataset_params import DatasetParams
from src.params.preprocessing_params import PreprocessingParams


def yield_tokens(data_iter, tokenizer, index):
    for _, sample in enumerate(data_iter):
        yield tokenizer(sample[0][index])


class Preprocessing:
    UNKNOWN_TOKEN = "<UNK>"
    PADDING_TOKEN = "<PAD>"
    BEGINNING_TOKEN = "<BOS>"
    END_TOKEN = "<EOS>"

    SPECIAL_TOKENS = [UNKNOWN_TOKEN, PADDING_TOKEN, BEGINNING_TOKEN, END_TOKEN]

    def __init__(self, data_iter, dataset_params: DatasetParams, preprocessing_params: PreprocessingParams):
        self.dataset_params: DatasetParams = dataset_params
        self.lang2tokenizer = {
            language: get_tokenizer("spacy", language=preprocessing_params.spacy_tokenizer[language])
            for language in [
                dataset_params.source_language,
                dataset_params.target_language,
            ]
        }

        self.lang2vocabs = self._get_lang2vocabs(data_iter)

    def _get_vocab(self, data_iter, language, language_index, min_freq=2) -> Vocab:
        vocab = build_vocab_from_iterator(
            yield_tokens(data_iter, self.lang2tokenizer[language], language_index),
            specials=self.SPECIAL_TOKENS,
            min_freq=min_freq,
        )

        vocab.set_default_index(vocab[self.UNKNOWN_TOKEN])

        return vocab

    def _get_lang2vocabs(self, data_iter) -> dict:
        source_vocab = self._get_vocab(
            data_iter,
            self.dataset_params.source_language,
            self.dataset_params.source_language_index,
        )
        target_vocab = self._get_vocab(
            data_iter,
            self.dataset_params.target_language,
            self.dataset_params.target_language_index,
        )

        return {
            self.dataset_params.source_language: source_vocab,
            self.dataset_params.target_language: target_vocab,
        }

    def stoi(self, s: str, language: str) -> List[int]:
        return (
            [self.lang2vocabs[language][self.BEGINNING_TOKEN]]
            + [
                self.lang2vocabs[language][token]
                for token in self.lang2tokenizer[language](s)
            ]
            + [self.lang2vocabs[language][self.END_TOKEN]]
        )

    def itos(self, tensor: torch.Tensor, language: str) -> List[str]:
        text = [self.lang2vocabs[language].get_itos()[index] for index in tensor]
        return text[1:-1]

    def sample_transform(self, sample: Tuple[str, str],) -> Tuple[Tensor, Tensor]:
        source_tensor = torch.tensor(
            self.stoi(
                sample[self.dataset_params.source_language_index],
                self.dataset_params.source_language,
            ),
            dtype=torch.long,
        )

        target_tensor = torch.tensor(
            self.stoi(
                sample[self.dataset_params.target_language_index],
                self.dataset_params.target_language,
            ),
            dtype=torch.long,
        )

        return source_tensor, target_tensor
