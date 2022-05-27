import os.path
import pickle

import torch
from torch import nn
import torchtext
import logging
import fasttext
import fasttext.util
from tqdm import tqdm

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m: nn.Module) -> None:
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


language_to_word2vec = {
    "ru": "../../data/fasttext/cc.ru.300.bin",
    "en": "../../data/fasttext/cc.en.300.bin",
}


def get_fasttext_pretrained_embedding(
    language, vocab: torchtext.vocab.Vocab, emb_dim: int
) -> nn.Module:
    assert emb_dim <= 300
    ft = fasttext.load_model(language_to_word2vec[language])
    if emb_dim < 300:
        fasttext.util.reduce_model(ft, emb_dim)

    emb_file_path = f"pretrained_emb_{language}_{emb_dim}.pkl"

    if not os.path.exists(emb_file_path):
        words_in_vocab = 0
        weights = torch.randn((len(vocab), emb_dim))
        for token, index in tqdm(vocab.get_stoi().items(), total=len(vocab)):
            if token in ft:
                weights[index] = torch.FloatTensor(ft.get_word_vector(token))
                words_in_vocab += 1

        embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        logger.info(
            f"Set pretrained embeddings for {words_in_vocab/len(vocab)}% of {language} words."
        )

        with open(emb_file_path, "wb") as f:
            pickle.dump(embedding, f)
    else:
        with open(emb_file_path, "rb") as f:
            embedding = pickle.load(f)

    return embedding
