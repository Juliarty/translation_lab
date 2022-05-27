from typing import List, Tuple

import torch.utils.data
import logging

from torch import Tensor
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from transformers import AutoTokenizer, BertTokenizer

from src.params.dataset_params import DatasetParams
from src.params.preprocessing_params import PreprocessingParams


logger = logging.getLogger(__name__)


def yield_tokens(data_iter, tokenizer, index):
    for _, sample in enumerate(data_iter):
        yield tokenizer(sample[0][index])


class BertEmbPreprocessing:
    UNKNOWN_TOKEN = "<UNK>"
    PADDING_TOKEN = "<PAD>"
    BEGINNING_TOKEN = "<BOS>"
    END_TOKEN = "<EOS>"

    SPECIAL_TOKENS = [UNKNOWN_TOKEN, PADDING_TOKEN, BEGINNING_TOKEN, END_TOKEN]

    def __init__(
        self,
        data_iter,
        dataset_params: DatasetParams,
        preprocessing_params: PreprocessingParams,
    ):
        self.dataset_params = dataset_params
        self.lang2tokenizer = {
            "ru": AutoTokenizer.from_pretrained(
                "../../deeppavlov/rubert_cased_L-12_H-768_A-12_pt"
            ),
            "en": get_tokenizer(
                "spacy", language=preprocessing_params.spacy_tokenizer["en"]
            ),
        }

        self.en_vocab = self._get_en_spacy_vocab(data_iter)

        self.VOCAB_SRC_SIZE = self.lang2tokenizer["ru"].vocab_size
        self.VOCAB_TRG_SIZE = len(self.en_vocab)
        self.PADDING_TRG_TOKEN_ID = self.en_vocab[self.PADDING_TOKEN]
        self.PADDING_SRC_TOKEN_ID = self.lang2tokenizer["ru"].pad_token_id

    def _get_en_spacy_vocab(self, data_iter):
        vocab = build_vocab_from_iterator(
            yield_tokens(data_iter, self.lang2tokenizer["en"], 0),
            specials=self.SPECIAL_TOKENS,
            min_freq=1,
        )
        vocab.set_default_index(vocab[self.UNKNOWN_TOKEN])
        return vocab

    def stoi(self, s: str, language: str) -> Tensor:
        if language == "ru":
            return self.lang2tokenizer["ru"](s, return_tensors="pt").input_ids
        elif language == "en":
            return torch.tensor(
                (
                    [self.en_vocab[self.BEGINNING_TOKEN]]
                    + [
                        self.en_vocab[token]
                        for token in self.lang2tokenizer[language](s)
                    ]
                    + [self.en_vocab[self.END_TOKEN]]
                ),
                dtype=torch.long,
            )
        else:
            raise Exception()

    def itos(self, tensor: torch.Tensor, language: str) -> List[str]:
        if language == "en":
            text = [self.en_vocab.get_itos()[index] for index in tensor]
            return text[1:-1]
        elif language == "ru":
            return self.lang2tokenizer[language].decode(tensor)

    def sample_transform(self, sample: Tuple[str, str],) -> Tuple[Tensor, Tensor]:
        source_tensor = self.stoi(
            sample[self.dataset_params.source_language_index],
            self.dataset_params.source_language,
        )

        target_tensor = self.stoi(
            sample[self.dataset_params.target_language_index],
            self.dataset_params.target_language,
        )

        return source_tensor, target_tensor


class Preprocessing:
    UNKNOWN_TOKEN = "<UNK>"
    PADDING_TOKEN = "<PAD>"
    BEGINNING_TOKEN = "<BOS>"
    END_TOKEN = "<EOS>"

    SPECIAL_TOKENS = [UNKNOWN_TOKEN, PADDING_TOKEN, BEGINNING_TOKEN, END_TOKEN]

    def __init__(
        self,
        data_iter,
        dataset_params: DatasetParams,
        preprocessing_params: PreprocessingParams,
    ):
        self.dataset_params: DatasetParams = dataset_params
        self.lang2tokenizer = {
            language: get_tokenizer(
                "spacy", language=preprocessing_params.spacy_tokenizer[language]
            )
            for language in [
                dataset_params.source_language,
                dataset_params.target_language,
            ]
        }

        self.lang2vocabs = self._get_lang2vocabs(data_iter)
        self.VOCAB_SRC_SIZE = len(self.lang2vocabs[dataset_params.source_language])
        self.VOCAB_TRG_SIZE = len(self.lang2vocabs[dataset_params.target_language])
        self.PADDING_TRG_TOKEN_ID = self.lang2vocabs[dataset_params.target_language][
            self.PADDING_TOKEN
        ]
        self.PADDING_SRC_TOKEN_ID = self.lang2vocabs[dataset_params.source_language][
            self.PADDING_TOKEN
        ]

    def _get_vocab(self, data_iter, language, language_index, min_freq=2) -> Vocab:
        vocab = build_vocab_from_iterator(
            yield_tokens(data_iter, self.lang2tokenizer[language], language_index),
            specials=self.SPECIAL_TOKENS,
            min_freq=min_freq,
        )

        vocab.set_default_index(vocab[self.UNKNOWN_TOKEN])
        logger.info(f"Get vocab for {language}: {len(vocab)}")
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
