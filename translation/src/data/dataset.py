import math
import os
import logging
import random
from typing import Tuple

import pandas as pd
import requests
import torch.utils.data
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, SequentialSampler
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from src.data.preprocessing import Preprocessing, BertEmbPreprocessing
from src.params.dataset_params import DatasetParams
from src.params.preprocessing_params import PreprocessingParams

logger = logging.getLogger(__name__)


class TranslationDataset(Dataset):
    def __init__(self, dataset_path: str, preprocessing: Preprocessing = None):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path, sep="\t")
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.preprocessing is None:
            return self.df.iloc[idx, :].values
        else:
            return self.preprocessing.sample_transform(self.df.iloc[idx, :].values)


def load_data(dataset_path: str, dataset_url: str):
    logger.info("Loading dataset")
    if not os.path.exists(dataset_path):
        logger.info("Dataset not found locally. Downloading from github.")
        r = requests.get(dataset_url)

        with open(dataset_path, "wb") as f:
            f.write(r.content)
    logger.info(f"Dataset ahs been loaded to {os.path.abspath(dataset_path)}")


def split_data(dataset_params: DatasetParams) -> None:
    df = pd.read_csv(dataset_params.dataset_path, sep="\t")
    df = df.sample(frac=1)
    train_length = math.floor(dataset_params.train_size * len(df))
    val_length = math.floor(dataset_params.val_size * len(df))
    test_length = len(df) - (train_length + val_length)
    logger.info(f"Split data into: {train_length, val_length, test_length}")

    x, test_df = train_test_split(df, test_size=dataset_params.test_size)
    train_df, val_df = train_test_split(
        df,
        test_size=dataset_params.val_size
        / (dataset_params.val_size + dataset_params.train_size),
    )

    train_df = train_df.apply(lambda col: col.str.lower())
    val_df = val_df.apply(lambda col: col.str.lower())
    test_df = test_df.apply(lambda col: col.str.lower())

    train_df.to_csv(dataset_params.train_dataset_path, sep="\t", index=False)
    val_df.to_csv(dataset_params.val_dataset_path, sep="\t", index=False)
    test_df.to_csv(dataset_params.test_dataset_path, sep="\t", index=False)

    logger.info("Split dataset.")


def get_preprocessing(
    dataset_params: DatasetParams, preprocessing_params: PreprocessingParams
):
    train_ds = TranslationDataset(dataset_path=dataset_params.train_dataset_path)

    if preprocessing_params.bert_emb:
        return BertEmbPreprocessing(
            data_iter=DataLoader(train_ds, collate_fn=lambda x: x),
            dataset_params=dataset_params,
            preprocessing_params=preprocessing_params,
        )
    else:
        return Preprocessing(
            data_iter=DataLoader(train_ds, collate_fn=lambda x: x),
            dataset_params=dataset_params,
            preprocessing_params=preprocessing_params,
        )


def get_dataloaders(
    dataset_params: DatasetParams, preprocessing: Preprocessing, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = TranslationDataset(
        dataset_path=dataset_params.train_dataset_path, preprocessing=preprocessing
    )
    val_ds = TranslationDataset(
        dataset_path=dataset_params.val_dataset_path, preprocessing=preprocessing
    )
    test_ds = TranslationDataset(
        dataset_path=dataset_params.test_dataset_path, preprocessing=preprocessing
    )

    def collate_fn(data_batch):
        source_batch, target_batch = [], []
        for (src_item, tgt_item) in data_batch:
            source_batch.append(src_item.squeeze(0))
            target_batch.append(tgt_item.squeeze(0))

        source_batch = pad_sequence(
            source_batch, padding_value=preprocessing.PADDING_SRC_TOKEN_ID
        )
        target_batch = pad_sequence(
            target_batch, padding_value=preprocessing.PADDING_TRG_TOKEN_ID
        )
        return source_batch, target_batch

    def batch_sampler():
        indices = [
            (i, len(s[dataset_params.target_language_index]))
            for i, s in enumerate(DataLoader(train_ds, collate_fn=lambda x: x))
        ]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(indices), batch_size * 100):
            pooled_indices.extend(
                sorted(indices[i : i + batch_size * 100], key=lambda x: x[1])
            )

        pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        for i in range(0, len(pooled_indices), batch_size):
            yield pooled_indices[i : i + batch_size]

    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            # batch_sampler=batch_sampler()
        ),
        DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
    )
