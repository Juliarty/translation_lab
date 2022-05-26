import os
import pickle
from typing import Union
import hydra
import logging

import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import get_preprocessing, get_dataloaders, split_data, load_data
from src.models.seq2seq import Encoder, Attention, Decoder, Seq2Seq
from src.params.dataset_params import DatasetParams
from src.params.main_params import MainParams, get_main_params

from src.models.utils import count_parameters, get_fasttext_pretrained_embedding
from src.params.model_params import ModelParams
from src.train import train, evaluate_blue

logger = logging.getLogger(__name__)


def prepare_seq2seq_model(
    preprocessing,
    dataset_params: DatasetParams,
    model_params: ModelParams,
    device: torch.device,
) -> nn.Module:
    encoder_input_dim = len(preprocessing.lang2vocabs[dataset_params.source_language])
    decoder_input_dim = len(preprocessing.lang2vocabs[dataset_params.target_language])
    encoder_embedding = None

    if model_params.pretrained_embedding == 'fasttext':
        encoder_embedding = get_fasttext_pretrained_embedding(dataset_params.source_language, preprocessing.lang2vocabs[dataset_params.source_language], model_params.enc_emb_dim)

    enc = Encoder(
        rnn_type=model_params.rnn_type,
        bidirectional=model_params.bidirectional,
        input_dim=encoder_input_dim,
        emb_dim=model_params.enc_emb_dim,
        enc_hid_dim=model_params.enc_hid_dim,
        dec_hid_dim=model_params.dec_hid_dim,
        dropout=model_params.enc_dropout,
        device=device,
        pretrained_embedding=encoder_embedding

    )

    attn = Attention(
        model_params.enc_hid_dim,
        model_params.dec_hid_dim,
        model_params.attn_dim,
        device,
        encoder_bidirectional=model_params.bidirectional,
    )

    decoder_embedding = None
    if model_params.pretrained_embedding == 'fasttext':
        decoder_embedding = get_fasttext_pretrained_embedding(dataset_params.target_language, preprocessing.lang2vocabs[dataset_params.target_language],
                                                              model_params.dec_emb_dim)
    dec = Decoder(
        model_params.rnn_type,
        model_params.bidirectional,
        decoder_input_dim,
        model_params.dec_emb_dim,
        model_params.enc_hid_dim,
        model_params.dec_hid_dim,
        model_params.dec_dropout,
        attn,
        device,
        pretrained_embedding=decoder_embedding
    )

    model = Seq2Seq(enc, dec, device).to(device)

    return model


@hydra.main(version_base="1.1", config_path="config", config_name="main.yaml")
def start_training_pipeline(cfg: Union[DictConfig, MainParams]) -> None:
    if isinstance(cfg, MainParams):
        main_params = cfg
    else:
        main_params = get_main_params(dict(cfg))

    load_data(
        dataset_path=main_params.dataset.dataset_path,
        dataset_url=main_params.dataset.dataset_url,
    )
    split_data(main_params.dataset)

    preprocessing_path = "preprocessing.pkl"
    if not os.path.exists(preprocessing_path):
        with open(preprocessing_path, 'wb') as f:
            preprocessing = get_preprocessing(main_params.dataset, main_params.preprocessing)
            pickle.dump(preprocessing, f)
    else:
        with open("preprocessing.pkl", 'rb') as f:
            preprocessing = pickle.load(f)

    train_dl, val_dl, test_dl = get_dataloaders(
        main_params.dataset, preprocessing, main_params.train.batch_size
    )

    model = prepare_seq2seq_model(
        preprocessing, main_params.dataset, main_params.model, main_params.train.device
    )
    logger.info(f"The model has {count_parameters(model):,} trainable parameters")
    optimizer = optim.Adam(model.parameters(), lr=main_params.train.learning_rate)
    criterion = nn.CrossEntropyLoss(
        ignore_index=preprocessing.lang2vocabs[main_params.dataset.target_language][
            preprocessing.PADDING_TOKEN
        ]
    )

    logger.info("Start training.")
    writer = SummaryWriter(main_params.tensorboard_logdir)

    model = train(
        model,
        train_dl=train_dl,
        valid_iter=val_dl,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=main_params.train.n_epoch,
        clip=1,
        device=main_params.train.device,
        summary_writer=writer,
        debug=main_params.debug,
        tmp_model_save_path=os.path.join(main_params.output_path, "tmp_model.pt"),
    )

    score = evaluate_blue(
        model=model,
        preprocessing=preprocessing,
        dataset_params=main_params.dataset,
        test_iterator=test_dl,
        device=main_params.train.device,
        summary_writer=writer,
        debug=main_params.debug,
    )

    logger.info(f"BLEU: {score}")
    logger.info(f"Saving results")

    with open(os.path.join(".", "model.pt"), "wb") as f:
        torch.save(model, f)


if __name__ == "__main__":
    start_training_pipeline()
